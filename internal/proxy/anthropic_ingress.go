package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/server"
	"github.com/ShubhamDX/aion/internal/telemetry"
	"github.com/ShubhamDX/aion/internal/types"
)

// ---------- Anthropic ingress request/response types ----------

// anthropicIngressRequest mirrors the Anthropic Messages API request schema.
type anthropicIngressRequest struct {
	Model         string                 `json:"model"`
	Messages      []anthropicIngressMsg  `json:"messages"`
	System        json.RawMessage        `json:"system,omitempty"`
	MaxTokens     int                    `json:"max_tokens"`
	Stream        bool                   `json:"stream,omitempty"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	StopSequences json.RawMessage        `json:"stop_sequences,omitempty"`
	Tools         []anthropicIngressTool `json:"tools,omitempty"`
	Metadata      json.RawMessage        `json:"metadata,omitempty"`
}

type anthropicIngressMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or []content-block
}

type anthropicIngressTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// anthropicIngressResponse is the Anthropic Messages API non-streaming response.
type anthropicIngressResponse struct {
	ID           string                         `json:"id"`
	Type         string                         `json:"type"` // "message"
	Role         string                         `json:"role"` // "assistant"
	Content      []anthropicIngressContentBlock `json:"content"`
	Model        string                         `json:"model"`
	StopReason   string                         `json:"stop_reason"`
	StopSequence *string                        `json:"stop_sequence"`
	Usage        anthropicIngressUsage          `json:"usage"`
}

type anthropicIngressContentBlock struct {
	Type  string          `json:"type"` // "text" or "tool_use"
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`    // tool_use
	Name  string          `json:"name,omitempty"`  // tool_use
	Input json.RawMessage `json:"input,omitempty"` // tool_use
}

type anthropicIngressUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

type anthropicIngressError struct {
	Type  string                    `json:"type"` // "error"
	Error anthropicIngressErrDetail `json:"error"`
}

type anthropicIngressErrDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// ---------- translation: Anthropic request -> OpenAI internal ----------

func translateAnthropicToOpenAI(aReq *anthropicIngressRequest) *types.ChatCompletionRequest {
	oReq := &types.ChatCompletionRequest{
		Model:       aReq.Model,
		Stream:      aReq.Stream,
		Temperature: aReq.Temperature,
		TopP:        aReq.TopP,
	}

	if aReq.MaxTokens > 0 {
		mt := aReq.MaxTokens
		oReq.MaxTokens = &mt
	}

	// Map stop_sequences -> Stop.
	if len(aReq.StopSequences) > 0 {
		oReq.Stop = aReq.StopSequences
	}

	// Extract system prompt. The system field can be a string or an array of
	// content blocks. We handle both.
	if len(aReq.System) > 0 {
		var sysStr string
		if err := json.Unmarshal(aReq.System, &sysStr); err == nil && sysStr != "" {
			b, _ := json.Marshal(sysStr)
			oReq.Messages = append(oReq.Messages, types.Message{
				Role:    "system",
				Content: b,
			})
		} else {
			// Try array of {type:"text",text:"..."} blocks.
			var blocks []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if err := json.Unmarshal(aReq.System, &blocks); err == nil {
				var combined string
				for _, b := range blocks {
					if b.Type == "text" {
						combined += b.Text
					}
				}
				if combined != "" {
					b, _ := json.Marshal(combined)
					oReq.Messages = append(oReq.Messages, types.Message{
						Role:    "system",
						Content: b,
					})
				}
			}
		}
	}

	// Convert messages. Anthropic content can be a string or array of blocks.
	for _, m := range aReq.Messages {
		msg := types.Message{
			Role:    m.Role,
			Content: m.Content,
		}

		// If content is an array of blocks, check for tool_result blocks
		// and convert them to OpenAI tool role messages.
		var blocks []json.RawMessage
		if json.Unmarshal(m.Content, &blocks) == nil && len(blocks) > 0 {
			// Check if this is a tool_result message.
			var firstBlock struct {
				Type      string          `json:"type"`
				ToolUseID string          `json:"tool_use_id"`
				Content   json.RawMessage `json:"content"`
			}
			if json.Unmarshal(blocks[0], &firstBlock) == nil && firstBlock.Type == "tool_result" {
				// Each tool_result block becomes a separate tool-role message.
				for _, raw := range blocks {
					var tr struct {
						Type      string          `json:"type"`
						ToolUseID string          `json:"tool_use_id"`
						Content   json.RawMessage `json:"content"`
					}
					if json.Unmarshal(raw, &tr) == nil && tr.Type == "tool_result" {
						content := extractToolResultContent(tr.Content)
						b, _ := json.Marshal(content)
						oReq.Messages = append(oReq.Messages, types.Message{
							Role:       "tool",
							Content:    b,
							ToolCallID: tr.ToolUseID,
						})
					}
				}
				continue
			}

			// Check for text + tool_use mixed blocks (assistant message).
			// Extract text and tool_calls.
			var textContent string
			var toolCalls []types.ToolCall
			for _, raw := range blocks {
				var block struct {
					Type  string          `json:"type"`
					Text  string          `json:"text"`
					ID    string          `json:"id"`
					Name  string          `json:"name"`
					Input json.RawMessage `json:"input"`
				}
				if json.Unmarshal(raw, &block) != nil {
					continue
				}
				switch block.Type {
				case "text":
					textContent += block.Text
				case "tool_use":
					toolCalls = append(toolCalls, types.ToolCall{
						ID:   block.ID,
						Type: "function",
						Function: types.FunctionCall{
							Name:      block.Name,
							Arguments: string(block.Input),
						},
					})
				}
			}
			if len(toolCalls) > 0 {
				b, _ := json.Marshal(textContent)
				oReq.Messages = append(oReq.Messages, types.Message{
					Role:      m.Role,
					Content:   b,
					ToolCalls: toolCalls,
				})
				continue
			}

			// Otherwise extract concatenated text for plain content blocks.
			var allText string
			isTextBlocks := true
			for _, raw := range blocks {
				var block struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}
				if json.Unmarshal(raw, &block) != nil || block.Type != "text" {
					isTextBlocks = false
					break
				}
				allText += block.Text
			}
			if isTextBlocks && allText != "" {
				b, _ := json.Marshal(allText)
				msg.Content = b
			}
		}

		oReq.Messages = append(oReq.Messages, msg)
	}

	// Convert tools: input_schema -> parameters.
	for _, t := range aReq.Tools {
		oReq.Tools = append(oReq.Tools, types.Tool{
			Type: "function",
			Function: types.FunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}

	return oReq
}

// extractToolResultContent extracts a string from a tool_result content field,
// which can be a string or an array of {type:"text",text:"..."} blocks.
func extractToolResultContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) == nil {
		var combined string
		for _, b := range blocks {
			if b.Type == "text" {
				combined += b.Text
			}
		}
		return combined
	}
	return string(raw)
}

// ---------- translation: OpenAI internal -> Anthropic response ----------

func translateOpenAIToAnthropic(resp *types.ChatCompletionResponse) *anthropicIngressResponse {
	aResp := &anthropicIngressResponse{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: resp.Model,
		Usage: anthropicIngressUsage{
			InputTokens:              resp.Usage.UncachedInputTokens,
			OutputTokens:             resp.Usage.CompletionTokens,
			CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     resp.Usage.CacheReadInputTokens,
		},
	}
	if aResp.Usage.InputTokens == 0 && aResp.Usage.CacheCreationInputTokens == 0 && aResp.Usage.CacheReadInputTokens == 0 {
		aResp.Usage.InputTokens = resp.Usage.PromptTokens
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		aResp.StopReason = mapOpenAIFinishReason(choice.FinishReason)

		// Convert text content.
		textContent := choice.Message.ContentString()
		if textContent != "" {
			aResp.Content = append(aResp.Content, anthropicIngressContentBlock{
				Type: "text",
				Text: textContent,
			})
		}

		// Convert tool calls -> tool_use blocks.
		for _, tc := range choice.Message.ToolCalls {
			aResp.Content = append(aResp.Content, anthropicIngressContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			})
		}
	}

	// Ensure content is never nil.
	if aResp.Content == nil {
		aResp.Content = []anthropicIngressContentBlock{}
	}

	return aResp
}

func mapOpenAIFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	default:
		return reason
	}
}

// ---------- handler ----------

// AnthropicMessages handles POST /v1/messages — the Anthropic-compatible ingress.
func (h *Handler) AnthropicMessages(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()
	requestID := server.RequestID(ctx)
	keyInfo := server.APIKeyInfo(ctx)

	// 1. Parse Anthropic request.
	var aReq anthropicIngressRequest
	if err := json.NewDecoder(r.Body).Decode(&aReq); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error",
			"Failed to parse request body: "+err.Error())
		return
	}

	if aReq.MaxTokens == 0 {
		aReq.MaxTokens = 4096
	}

	// 2. Translate to internal OpenAI format.
	req := translateAnthropicToOpenAI(&aReq)

	// 3. Determine routing (same logic as ChatCompletion).
	var (
		selectedModel *router.ModelOption
		tier          types.Tier
		score         float64
		signals       map[string]float64
	)

	model := req.Model
	switch {
	case model == "" || model == "aion-auto":
		tier, score, signals = h.classifier.Classify(req)
		var err error
		selectedModel, err = h.router.RouteWithFallback(tier)
		if err != nil {
			writeAnthropicError(w, http.StatusServiceUnavailable, "api_error",
				"No healthy model available for tier: "+err.Error())
			return
		}

	case model == "aion-local":
		tier = types.Tier1
		var err error
		selectedModel, err = h.router.FindByProvider("local")
		if err != nil {
			writeAnthropicError(w, http.StatusServiceUnavailable, "api_error",
				"No local model available: "+err.Error())
			return
		}

	case model == "aion-escalate":
		tier = types.Tier3
		var err error
		selectedModel, err = h.router.RouteWithFallback(types.Tier3)
		if err != nil {
			writeAnthropicError(w, http.StatusServiceUnavailable, "api_error", err.Error())
			return
		}

	default:
		var err error
		selectedModel, err = h.router.FindModel(model)
		if err != nil {
			// Model not found — fall back to auto-classification rather
			// than rejecting. Anthropic SDK clients often send model names
			// that don't match AION's configured model IDs exactly.
			slog.Info("model not found, falling back to auto-routing",
				"requested_model", model, "request_id", requestID)
			tier, score, signals = h.classifier.Classify(req)
			selectedModel, err = h.router.RouteWithFallback(tier)
			if err != nil {
				writeAnthropicError(w, http.StatusServiceUnavailable, "api_error",
					"No healthy model available: "+err.Error())
				return
			}
		}
		tier = selectedModel.Tier
	}

	// 3b. Gateway pre-request hook (optional) — SAME enterprise policy path as
	// the OpenAI ingress, so /v1/messages traffic cannot bypass decisions or
	// signed evidence. A nil hook leaves behavior unchanged.
	sessionMaterial := types.SessionMaterialFromRequest(req, r.Header.Get(sessionIDHeader))
	// Material extracted; scrub the raw session id so it never reaches the hook
	// (digests only) or an upstream provider.
	types.ScrubSessionID(req)
	if h.hooks != nil && h.hooks.PreRequest != nil {
		dec := h.hooks.PreRequest(types.PreRequestInput{
			RequestID:       requestID,
			PrincipalID:     keyIDFromInfo(keyInfo),
			Request:         req,
			RequestedModel:  model,
			RoutedModel:     selectedModel.ID,
			RoutedProvider:  selectedModel.Provider,
			Tier:            tier,
			EstimatedCost:   h.estimatedCost(req, selectedModel),
			RequestDigest:   types.RequestContentDigest(req),
			SessionMaterial: sessionMaterial,
		})
		switch dec.Verdict {
		case types.VerdictBlock:
			w.Header().Set("X-AION-Decision", "block")
			writeAnthropicError(w, http.StatusForbidden, "permission_error", decisionMessage(dec, "request blocked by policy"))
			return
		case types.VerdictHold:
			w.Header().Set("X-AION-Decision", "hold")
			writeAnthropicError(w, http.StatusAccepted, "approval_required", decisionMessage(dec, "request held for approval"))
			return
		case types.VerdictRoute:
			m, changed, err := h.resolveRouteOverride(dec, selectedModel)
			if err != nil {
				w.Header().Set("X-AION-Decision", "route_error")
				writeAnthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
				return
			}
			if changed {
				selectedModel = m
				tier = m.Tier
				w.Header().Set("X-AION-Decision", "route")
			}
		}
	}

	// 4. Budget check.
	if keyInfo != nil && h.budget != nil {
		if err := h.budget.Check(ctx, keyInfo.Key, keyInfo.DailyLimitUSD, keyInfo.MonthlyLimitUSD); err != nil {
			writeAnthropicError(w, http.StatusTooManyRequests, "rate_limit_error", err.Error())
			return
		}
	}

	// 5. Resolve provider.
	prov, ok := h.registry.Get(selectedModel.Provider)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error",
			"Provider not found: "+selectedModel.Provider)
		return
	}

	// 6. Log routing decision and set AION response headers.
	slog.Info("routed",
		"request_id", requestID,
		"ingress", "anthropic",
		"requested_model", model,
		"routed_model", selectedModel.ID,
		"provider", selectedModel.Provider,
		"tier", int(tier),
		"score", fmt.Sprintf("%.3f", score),
		"signals", signals,
		"stream", req.Stream,
	)
	w.Header().Set("X-AION-Tier", fmt.Sprintf("%d", int(tier)))
	w.Header().Set("X-AION-Model", selectedModel.ID)
	w.Header().Set("X-AION-Provider", selectedModel.Provider)
	w.Header().Set("X-Request-ID", requestID)

	// 7. Dispatch — streaming or non-streaming.
	if req.Stream {
		h.handleAnthropicStream(w, r, req, prov, selectedModel, tier, score, signals, keyInfo, requestID, start, sessionMaterial)
		return
	}

	// Non-streaming path.
	resp, err := prov.Send(ctx, req, selectedModel.ID)
	if err != nil {
		slog.Error("provider error",
			"provider", selectedModel.Provider,
			"model", selectedModel.ID,
			"error", err,
		)
		writeAnthropicError(w, http.StatusBadGateway, "api_error",
			"Provider request failed: "+err.Error())
		return
	}

	// Calculate cost and savings.
	var costUSD, savingsUSD float64
	var costBreakdown pricing.CostBreakdown
	if resp.ChatResponse != nil && resp.ChatResponse.Usage.TotalTokens > 0 {
		costBreakdown, savingsUSD = h.costAndSavings(selectedModel.ID, resp.ChatResponse.Usage)
		costUSD = costBreakdown.TotalUSD
	}

	w.Header().Set("X-AION-Cost-USD", fmt.Sprintf("%.6f", costUSD))
	w.Header().Set("X-AION-Savings-USD", fmt.Sprintf("%.6f", savingsUSD))

	// Record telemetry.
	if h.recorder != nil && resp.ChatResponse != nil {
		h.recorder.Record(telemetry.RequestEvent{
			RequestID:    requestID,
			APIKeyID:     keyIDFromInfo(keyInfo),
			Tier:         int(tier),
			Model:        selectedModel.ID,
			Provider:     selectedModel.Provider,
			InputTokens:  resp.ChatResponse.Usage.PromptTokens,
			OutputTokens: resp.ChatResponse.Usage.CompletionTokens,
			CostUSD:      costUSD,
			SavingsUSD:   savingsUSD,
			LatencyMS:    time.Since(start).Milliseconds(),
			StatusCode:   resp.StatusCode,
			Stream:       false,
			CreatedAt:    time.Now().UTC(),
		})
	}

	// Record budget spend.
	if keyInfo != nil && h.budget != nil && costUSD > 0 {
		_ = h.budget.Record(ctx, keyInfo.Key, costUSD)
	}

	// Write response FIRST (same ordering as the OpenAI path): the customer
	// response is fully served before the post-response hook runs, so the hook's
	// evidence recording / response-shape validation can never delay or block the
	// served response.
	aResp := translateOpenAIToAnthropic(resp.ChatResponse)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(aResp)

	// Gateway post-response hook (optional): record signed evidence for the
	// Anthropic-ingress request, same as the OpenAI path. Dispatched
	// ASYNCHRONOUSLY (after the response is written) so the hook cannot delay the
	// served response, even via net/http response buffering before handler return.
	if h.hooks != nil && h.hooks.PostResponse != nil && resp.ChatResponse != nil {
		postMaterial := sessionMaterial
		postMaterial.NextCachePrefixMaterialSHA256 = types.NextCachePrefixMaterial(req, resp.ChatResponse)
		h.dispatchPostResponse(postResponseInputWithUsage(types.PostResponseInput{
			RequestID:        requestID,
			PrincipalID:      keyIDFromInfo(keyInfo),
			RequestedModel:   model,
			RoutedModel:      selectedModel.ID,
			RoutedProvider:   selectedModel.Provider,
			Tier:             tier,
			InputTokens:      resp.ChatResponse.Usage.PromptTokens,
			OutputTokens:     resp.ChatResponse.Usage.CompletionTokens,
			CostUSD:          costUSD,
			SavingsUSD:       savingsUSD,
			LatencyMS:        time.Since(start).Milliseconds(),
			StatusCode:       resp.StatusCode,
			Stream:           false,
			RequestDigest:    types.RequestContentDigest(req),
			ResponseDigest:   types.ResponseContentDigest(resp.ChatResponse),
			SessionMaterial:  postMaterial,
			ResponseContents: types.ResponseContentStrings(resp.ChatResponse),
		}, resp.ChatResponse.Usage, costBreakdown))
	}
}

// ---------- streaming ----------

// handleAnthropicStream reads OpenAI SSE chunks from a provider and emits
// Anthropic-format SSE events to the client.
func (h *Handler) handleAnthropicStream(
	w http.ResponseWriter,
	r *http.Request,
	req *types.ChatCompletionRequest,
	prov provider.Provider,
	model *router.ModelOption,
	tier types.Tier,
	score float64,
	signals map[string]float64,
	keyInfo *apikey.KeyInfo,
	requestID string,
	start time.Time,
	sessionMaterial types.SessionMaterial,
) {
	ctx := r.Context()

	stream, err := prov.SendStream(ctx, req, model.ID)
	if err != nil {
		slog.Error("stream error",
			"provider", model.Provider,
			"model", model.ID,
			"error", err,
		)
		writeAnthropicError(w, http.StatusBadGateway, "api_error",
			"Stream request failed: "+err.Error())
		return
	}
	defer stream.Close()

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error",
			"Streaming not supported")
		return
	}

	// Set SSE headers.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	// Emit message_start event.
	msgStart := map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":      requestID,
			"type":    "message",
			"role":    "assistant",
			"content": []interface{}{},
			"model":   model.ID,
			"usage": map[string]int{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	}
	writeSSEEvent(w, flusher, "message_start", msgStart)

	// Track state for building Anthropic events from OpenAI chunks.
	contentBlockStarted := false
	blockIndex := 0
	var totalUsage types.Usage
	var lastFinishReason string

	for {
		chunk, err := stream.ReadChunk()
		if err != nil {
			if err == io.EOF {
				break
			}
			slog.Error("stream read error", "error", err)
			break
		}

		if chunk.Usage != nil {
			totalUsage.MergeFrom(*chunk.Usage)
		}

		for _, choice := range chunk.Choices {
			// Track finish reason.
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				lastFinishReason = *choice.FinishReason
				continue
			}

			// Content delta.
			if choice.Delta.Content != nil && *choice.Delta.Content != "" {
				if !contentBlockStarted {
					// Emit content_block_start.
					cbStart := map[string]interface{}{
						"type":  "content_block_start",
						"index": blockIndex,
						"content_block": map[string]interface{}{
							"type": "text",
							"text": "",
						},
					}
					writeSSEEvent(w, flusher, "content_block_start", cbStart)
					contentBlockStarted = true
				}

				// Emit content_block_delta.
				cbDelta := map[string]interface{}{
					"type":  "content_block_delta",
					"index": blockIndex,
					"delta": map[string]interface{}{
						"type": "text_delta",
						"text": *choice.Delta.Content,
					},
				}
				writeSSEEvent(w, flusher, "content_block_delta", cbDelta)
			}

			// Tool call deltas — emit as tool_use blocks.
			for _, tc := range choice.Delta.ToolCalls {
				if contentBlockStarted {
					cbStop := map[string]interface{}{
						"type":  "content_block_stop",
						"index": blockIndex,
					}
					writeSSEEvent(w, flusher, "content_block_stop", cbStop)
					blockIndex++
					contentBlockStarted = false
				}

				cbStart := map[string]interface{}{
					"type":  "content_block_start",
					"index": blockIndex,
					"content_block": map[string]interface{}{
						"type":  "tool_use",
						"id":    tc.ID,
						"name":  tc.Function.Name,
						"input": map[string]interface{}{},
					},
				}
				writeSSEEvent(w, flusher, "content_block_start", cbStart)

				if tc.Function.Arguments != "" {
					cbDelta := map[string]interface{}{
						"type":  "content_block_delta",
						"index": blockIndex,
						"delta": map[string]interface{}{
							"type":         "input_json_delta",
							"partial_json": tc.Function.Arguments,
						},
					}
					writeSSEEvent(w, flusher, "content_block_delta", cbDelta)
				}

				cbStop := map[string]interface{}{
					"type":  "content_block_stop",
					"index": blockIndex,
				}
				writeSSEEvent(w, flusher, "content_block_stop", cbStop)
				blockIndex++
			}
		}
	}

	// Close any open content block.
	if contentBlockStarted {
		cbStop := map[string]interface{}{
			"type":  "content_block_stop",
			"index": blockIndex,
		}
		writeSSEEvent(w, flusher, "content_block_stop", cbStop)
	}

	// Emit message_delta with stop_reason.
	stopReason := mapOpenAIFinishReason(lastFinishReason)
	if stopReason == "" {
		stopReason = "end_turn"
	}
	msgDelta := map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]int{
			"output_tokens": totalUsage.CompletionTokens,
		},
	}
	writeSSEEvent(w, flusher, "message_delta", msgDelta)

	// Emit message_stop.
	writeSSEEvent(w, flusher, "message_stop", map[string]string{"type": "message_stop"})

	// Calculate cost and savings.
	var costUSD, savingsUSD float64
	var costBreakdown pricing.CostBreakdown
	if totalUsage.TotalTokens > 0 {
		costBreakdown, savingsUSD = h.costAndSavings(model.ID, totalUsage)
		costUSD = costBreakdown.TotalUSD
	}

	// Record telemetry.
	if h.recorder != nil {
		h.recorder.Record(telemetry.RequestEvent{
			RequestID:    requestID,
			APIKeyID:     keyIDFromInfo(keyInfo),
			Tier:         int(tier),
			Model:        model.ID,
			Provider:     model.Provider,
			InputTokens:  totalUsage.PromptTokens,
			OutputTokens: totalUsage.CompletionTokens,
			CostUSD:      costUSD,
			SavingsUSD:   savingsUSD,
			LatencyMS:    time.Since(start).Milliseconds(),
			StatusCode:   http.StatusOK,
			Stream:       true,
			CreatedAt:    time.Now().UTC(),
		})
	}

	// Record budget spend.
	if keyInfo != nil && h.budget != nil && costUSD > 0 {
		_ = h.budget.Record(ctx, keyInfo.Key, costUSD)
	}

	// Gateway post-response hook (optional). Streamed content is not reassembled,
	// so the output digest is empty (correlation-only output anchor).
	if h.hooks != nil && h.hooks.PostResponse != nil {
		// Streaming: next-turn prefix digest stays "" (stream not buffered).
		h.hooks.PostResponse(postResponseInputWithUsage(types.PostResponseInput{
			RequestID:       requestID,
			PrincipalID:     keyIDFromInfo(keyInfo),
			RequestedModel:  req.Model,
			RoutedModel:     model.ID,
			RoutedProvider:  model.Provider,
			Tier:            tier,
			InputTokens:     totalUsage.PromptTokens,
			OutputTokens:    totalUsage.CompletionTokens,
			CostUSD:         costUSD,
			SavingsUSD:      savingsUSD,
			LatencyMS:       time.Since(start).Milliseconds(),
			StatusCode:      http.StatusOK,
			Stream:          true,
			RequestDigest:   types.RequestContentDigest(req),
			ResponseDigest:  "",
			SessionMaterial: sessionMaterial,
		}, totalUsage, costBreakdown))
	}
}

// ---------- helpers ----------

// writeSSEEvent writes a single Anthropic SSE event.
func writeSSEEvent(w http.ResponseWriter, flusher http.Flusher, eventType string, data interface{}) {
	b, err := json.Marshal(data)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, b)
	flusher.Flush()
}

// writeAnthropicError writes an Anthropic-format JSON error response.
func writeAnthropicError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(anthropicIngressError{
		Type: "error",
		Error: anthropicIngressErrDetail{
			Type:    errType,
			Message: message,
		},
	})
}
