package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ShubhamDX/aion/internal/types"
	pkgtypes "github.com/ShubhamDX/aion/pkg/types"
)

func isBedrockNovaModel(model string) bool {
	for _, prefix := range []string{"amazon.nova-", "us.amazon.nova-", "eu.amazon.nova-", "apac.amazon.nova-", "global.amazon.nova-"} {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// Unsupported contracts fail explicitly rather than silently dropping the field.
// Native structured output remains available on the existing Mantle path.
func bedrockFormatSupported(req *types.ChatCompletionRequest) error {
	if req.ResponseFormat != nil && req.ResponseFormat.Type != "" && req.ResponseFormat.Type != "text" {
		return fmt.Errorf("bedrock: response_format is not supported on this model transport")
	}
	if req.SchemaSettings != nil && req.SchemaSettings.MustEmitNative {
		return fmt.Errorf("bedrock: mandatory native schema is not supported on this model transport")
	}
	return nil
}

func novaText(message types.Message) ([]map[string]any, error) {
	if len(message.Content) == 0 || string(message.Content) == "null" {
		return nil, nil
	}
	var text string
	if json.Unmarshal(message.Content, &text) == nil {
		if text == "" {
			return nil, nil
		}
		return []map[string]any{{"text": text}}, nil
	}
	var parts []struct {
		Type         string          `json:"type"`
		Text         string          `json:"text"`
		CacheControl json.RawMessage `json:"cache_control"`
	}
	if json.Unmarshal(message.Content, &parts) != nil {
		return nil, fmt.Errorf("bedrock Nova: expected text content")
	}
	var result []map[string]any
	for _, part := range parts {
		if part.Type != "text" {
			return nil, fmt.Errorf("bedrock Nova: unsupported content type %q", part.Type)
		}
		if part.Text != "" {
			result = append(result, map[string]any{"text": part.Text})
		}
		if len(part.CacheControl) > 0 && string(part.CacheControl) != "null" {
			var c struct {
				Type string `json:"type"`
				TTL  string `json:"ttl"`
			}
			if json.Unmarshal(part.CacheControl, &c) != nil || c.Type != "ephemeral" || (c.TTL != "" && c.TTL != "5m") {
				return nil, fmt.Errorf("bedrock Nova: unsupported cache control")
			}
			if message.Role == "assistant" || message.Role == "tool" {
				return nil, fmt.Errorf("bedrock Nova: cache checkpoints require system or user text")
			}
			result = append(result, map[string]any{"cachePoint": map[string]string{"type": "default"}})
		}
	}
	return result, nil
}

func novaRequest(req *types.ChatCompletionRequest) (map[string]any, error) {
	if err := bedrockFormatSupported(req); err != nil {
		return nil, err
	}
	maxTokens := bedrockDefaultMaxTok
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	inf := map[string]any{"maxTokens": maxTokens}
	if req.Temperature != nil {
		inf["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		inf["topP"] = *req.TopP
	}
	if len(req.Stop) > 0 && string(req.Stop) != "null" {
		var stops []string
		var one string
		if json.Unmarshal(req.Stop, &one) == nil {
			stops = []string{one}
		} else if json.Unmarshal(req.Stop, &stops) != nil {
			return nil, fmt.Errorf("bedrock Nova: invalid stop sequences")
		}
		inf["stopSequences"] = stops
	}
	var system []map[string]any
	messages := []map[string]any{}
	for _, m := range req.Messages {
		blocks, err := novaText(m)
		if err != nil {
			return nil, err
		}
		role := m.Role
		switch role {
		case "system", "developer":
			system = append(system, blocks...)
			continue
		case "user", "assistant":
		case "tool":
			if m.ToolCallID == "" {
				return nil, fmt.Errorf("bedrock Nova: tool result requires tool_call_id")
			}
			role = "user"
			if len(blocks) == 0 {
				blocks = []map[string]any{{"text": ""}}
			}
			blocks = []map[string]any{{"toolResult": map[string]any{"toolUseId": m.ToolCallID, "content": blocks}}}
		default:
			return nil, fmt.Errorf("bedrock Nova: unsupported role %q", role)
		}
		for _, call := range m.ToolCalls {
			if role != "assistant" {
				return nil, fmt.Errorf("bedrock Nova: tool calls require assistant role")
			}
			var input map[string]json.RawMessage
			if json.Unmarshal([]byte(call.Function.Arguments), &input) != nil || input == nil {
				return nil, fmt.Errorf("bedrock Nova: tool arguments must be an object")
			}
			blocks = append(blocks, map[string]any{"toolUse": map[string]any{"toolUseId": call.ID, "name": call.Function.Name, "input": input}})
		}
		if len(blocks) == 0 {
			return nil, fmt.Errorf("bedrock Nova: empty message")
		}
		if len(messages) > 0 && messages[len(messages)-1]["role"] == role {
			last := messages[len(messages)-1]
			last["content"] = append(last["content"].([]map[string]any), blocks...)
		} else {
			messages = append(messages, map[string]any{"role": role, "content": blocks})
		}
	}
	body := map[string]any{"messages": messages, "inferenceConfig": inf}
	if len(system) > 0 {
		body["system"] = system
	}
	if len(req.Tools) > 0 {
		tools := []map[string]any{}
		for _, tool := range req.Tools {
			tools = append(tools, map[string]any{"toolSpec": map[string]any{"name": tool.Function.Name, "description": tool.Function.Description, "inputSchema": map[string]any{"json": tool.Function.Parameters}}})
		}
		cfg := map[string]any{"tools": tools}
		if len(req.ToolChoice) > 0 && string(req.ToolChoice) != "null" {
			var choice string
			if json.Unmarshal(req.ToolChoice, &choice) == nil {
				switch choice {
				case "auto":
					cfg["toolChoice"] = map[string]any{"auto": map[string]any{}}
				case "required":
					cfg["toolChoice"] = map[string]any{"any": map[string]any{}}
				default:
					return nil, fmt.Errorf("bedrock Nova: unsupported tool_choice")
				}
			} else {
				var selected struct {
					Type     string `json:"type"`
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				}
				if json.Unmarshal(req.ToolChoice, &selected) != nil || selected.Type != "function" || selected.Function.Name == "" {
					return nil, fmt.Errorf("bedrock Nova: invalid tool_choice")
				}
				cfg["toolChoice"] = map[string]any{"tool": map[string]string{"name": selected.Function.Name}}
			}
		}
		body["toolConfig"] = cfg
	} else if len(req.ToolChoice) > 0 && string(req.ToolChoice) != "null" {
		return nil, fmt.Errorf("bedrock Nova: tool_choice requires tools")
	}
	return body, nil
}

type novaUsage struct {
	Input      int `json:"inputTokens"`
	Output     int `json:"outputTokens"`
	Total      int `json:"totalTokens"`
	Read       int `json:"cacheReadInputTokens"`
	Write      int `json:"cacheWriteInputTokens"`
	ReadCount  int `json:"cacheReadInputTokenCount"`
	WriteCount int `json:"cacheWriteInputTokenCount"`
}

func (u novaUsage) normalized() (types.Usage, error) {
	if u.Read != 0 && u.ReadCount != 0 && u.Read != u.ReadCount || u.Write != 0 && u.WriteCount != 0 && u.Write != u.WriteCount {
		return types.Usage{}, fmt.Errorf("bedrock Nova: conflicting cache usage counters")
	}
	read, write := u.Read, u.Write
	if read == 0 {
		read = u.ReadCount
	}
	if write == 0 {
		write = u.WriteCount
	}
	total := u.Input + read + write + u.Output
	if u.Input < 0 || u.Output < 0 || read < 0 || write < 0 || u.Total != 0 && total != u.Total {
		return types.Usage{}, fmt.Errorf("bedrock Nova: inconsistent token usage")
	}
	usage := types.Usage{PromptTokens: u.Input + read + write, CompletionTokens: u.Output, TotalTokens: total, UncachedInputTokens: u.Input, CacheReadInputTokens: read, CacheCreationInputTokens: write}
	if read > 0 || write > 0 {
		usage.PromptTokensDetails = &pkgtypes.PromptTokensDetails{CachedTokens: read, CacheWriteTokens: write}
		usage.ProviderCacheMode = "explicit"
	}
	return usage, nil
}

func (p *BedrockProvider) sendNova(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	payload, err := novaRequest(req)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/model/"+url.PathEscape(model)+"/converse", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if err = p.authorize(ctx, request, body); err != nil {
		return nil, err
	}
	resp, err := p.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("bedrock Nova: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		detail, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock Nova: HTTP %d: %s", resp.StatusCode, detail)
	}
	var raw struct {
		Output struct {
			Message struct {
				Role    string            `json:"role"`
				Content []json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"output"`
		Usage      novaUsage `json:"usage"`
		StopReason string    `json:"stopReason"`
	}
	if err = json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("bedrock Nova: decode: %w", err)
	}
	usage, err := raw.Usage.normalized()
	if err != nil {
		return nil, err
	}
	var text strings.Builder
	message := types.Message{Role: "assistant"}
	for _, block := range raw.Output.Message.Content {
		var item map[string]json.RawMessage
		if json.Unmarshal(block, &item) != nil {
			return nil, fmt.Errorf("bedrock Nova: invalid output block")
		}
		if value, ok := item["text"]; ok {
			var s string
			if json.Unmarshal(value, &s) != nil {
				return nil, fmt.Errorf("bedrock Nova: invalid text")
			}
			text.WriteString(s)
		} else if value, ok := item["toolUse"]; ok {
			var tool struct {
				ID    string          `json:"toolUseId"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			}
			if json.Unmarshal(value, &tool) != nil {
				return nil, fmt.Errorf("bedrock Nova: invalid tool output")
			}
			message.ToolCalls = append(message.ToolCalls, types.ToolCall{ID: tool.ID, Type: "function", Function: types.FunctionCall{Name: tool.Name, Arguments: string(tool.Input)}})
		} else {
			return nil, fmt.Errorf("bedrock Nova: unsupported output block")
		}
	}
	message.Content, _ = json.Marshal(text.String())
	finish := "stop"
	switch raw.StopReason {
	case "max_tokens":
		finish = "length"
	case "tool_use":
		finish = "tool_calls"
	case "content_filtered", "guardrail_intervened":
		finish = "content_filter"
	}
	id := resp.Header.Get("X-Amzn-Requestid")
	if id == "" {
		id = fmt.Sprintf("nova-%d", time.Now().UnixNano())
	}
	return &Response{StatusCode: resp.StatusCode, ChatResponse: &types.ChatCompletionResponse{ID: id, Object: "chat.completion", Created: time.Now().Unix(), Model: model, Choices: []types.Choice{{Index: 0, Message: message, FinishReason: finish}}, Usage: usage}}, nil
}

func (p *BedrockProvider) streamNova(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	payload, err := novaRequest(req)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/model/"+url.PathEscape(model)+"/converse-stream", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if err = p.authorize(ctx, request, body); err != nil {
		return nil, err
	}
	resp, err := p.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("bedrock Nova stream: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		detail, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock Nova stream: HTTP %d: %s", resp.StatusCode, detail)
	}
	return &novaStreamReader{frames: bedrockStreamReader{reader: resp.Body, body: resp.Body}, model: model, id: resp.Header.Get("X-Amzn-Requestid"), tools: map[int]int{}}, nil
}

type novaStreamReader struct {
	frames    bedrockStreamReader
	model, id string
	tools     map[int]int
}

func (s *novaStreamReader) Close() error { return s.frames.Close() }

func (s *novaStreamReader) ReadChunk() (*types.ChatCompletionChunk, error) {
	for {
		headers, payload, err := s.frames.readFrame()
		if err != nil {
			return nil, err
		}
		if headers[":message-type"] == "exception" {
			return nil, fmt.Errorf("bedrock Nova stream: %s", headers[":exception-type"])
		}
		if headers[":message-type"] != "event" {
			continue
		}
		chunk := &types.ChatCompletionChunk{ID: s.id, Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: s.model}
		choice := types.ChunkChoice{Index: 0}
		switch headers[":event-type"] {
		case "messageStart":
			choice.Delta.Role = "assistant"
		case "contentBlockStart":
			var value struct {
				Index int `json:"contentBlockIndex"`
				Start struct {
					Tool *struct {
						ID   string `json:"toolUseId"`
						Name string `json:"name"`
					} `json:"toolUse"`
				} `json:"start"`
			}
			if err := json.Unmarshal(payload, &value); err != nil {
				return nil, err
			}
			if value.Start.Tool == nil {
				continue
			}
			index := len(s.tools)
			s.tools[value.Index] = index
			choice.Delta.ToolCalls = []types.ToolCall{{Index: &index, ID: value.Start.Tool.ID, Type: "function", Function: types.FunctionCall{Name: value.Start.Tool.Name}}}
		case "contentBlockDelta":
			var value struct {
				Index int `json:"contentBlockIndex"`
				Delta struct {
					Text *string `json:"text"`
					Tool *struct {
						Input string `json:"input"`
					} `json:"toolUse"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(payload, &value); err != nil {
				return nil, err
			}
			if value.Delta.Text != nil {
				choice.Delta.Content = value.Delta.Text
			} else if value.Delta.Tool != nil {
				index, ok := s.tools[value.Index]
				if !ok {
					return nil, fmt.Errorf("bedrock Nova stream: tool delta without start")
				}
				choice.Delta.ToolCalls = []types.ToolCall{{Index: &index, Function: types.FunctionCall{Arguments: value.Delta.Tool.Input}}}
			} else {
				return nil, fmt.Errorf("bedrock Nova stream: unsupported content delta")
			}
		case "contentBlockStop":
			continue
		case "messageStop":
			var value struct {
				Reason string `json:"stopReason"`
			}
			if err := json.Unmarshal(payload, &value); err != nil {
				return nil, err
			}
			reason := "stop"
			switch value.Reason {
			case "max_tokens":
				reason = "length"
			case "tool_use":
				reason = "tool_calls"
			case "content_filtered", "guardrail_intervened":
				reason = "content_filter"
			}
			choice.FinishReason = &reason
		case "metadata":
			var value struct {
				Usage novaUsage `json:"usage"`
			}
			if err := json.Unmarshal(payload, &value); err != nil {
				return nil, err
			}
			usage, err := value.Usage.normalized()
			if err != nil {
				return nil, err
			}
			chunk.Usage = &usage
			chunk.Choices = []types.ChunkChoice{}
			return chunk, nil
		default:
			return nil, fmt.Errorf("bedrock Nova stream: unsupported event %q", headers[":event-type"])
		}
		chunk.Choices = []types.ChunkChoice{choice}
		return chunk, nil
	}
}
