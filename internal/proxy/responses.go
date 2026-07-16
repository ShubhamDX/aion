package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ShubhamDX/aion/internal/types"
)

type responsesRequest struct {
	Model           string          `json:"model"`
	Input           any             `json:"input"`
	Instructions    string          `json:"instructions,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	Stream          bool            `json:"stream,omitempty"`
	User            string          `json:"user,omitempty"`
	Tools           []responsesTool `json:"tools,omitempty"`
	ToolChoice      json.RawMessage `json:"tool_choice,omitempty"`
}

type responsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

type responsesResponse struct {
	ID        string                 `json:"id"`
	Object    string                 `json:"object"`
	CreatedAt int64                  `json:"created_at"`
	Model     string                 `json:"model"`
	Output    []any                  `json:"output"`
	Usage     responsesUsage         `json:"usage"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type responsesOutput struct {
	ID      string                   `json:"id"`
	Type    string                   `json:"type"`
	Status  string                   `json:"status"`
	Role    string                   `json:"role"`
	Content []responsesOutputContent `json:"content"`
}

type responsesOutputContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type responsesFunctionCall struct {
	ID        string `json:"id,omitempty"`
	Type      string `json:"type"`
	Status    string `json:"status,omitempty"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// Responses handles the OpenAI Responses compatibility endpoint by translating
// text input to the existing chat-completions lifecycle. The routing, budget,
// context and evidence hooks therefore see the same governed path as
// /v1/chat/completions.
func (h *Handler) Responses(w http.ResponseWriter, r *http.Request) {
	var req responsesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "Failed to parse request body: "+err.Error())
		return
	}
	chatReq, err := responsesToChatRequest(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	body, err := json.Marshal(chatReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", err.Error())
		return
	}

	r2 := r.Clone(r.Context())
	r2.Body = io.NopCloser(bytes.NewReader(body))
	r2.ContentLength = int64(len(body))
	if req.Stream {
		streamWriter := newResponsesStreamWriter(w)
		h.ChatCompletion(streamWriter, r2)
		streamWriter.writeCapturedError()
		return
	}

	rec := newCaptureResponseWriter()
	h.ChatCompletion(rec, r2)

	for key, values := range rec.Header() {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	if rec.status >= 400 {
		w.WriteHeader(rec.status)
		_, _ = w.Write(rec.body.Bytes())
		return
	}

	var chat types.ChatCompletionResponse
	if err := json.Unmarshal(rec.body.Bytes(), &chat); err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", "failed to translate chat response")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(rec.status)
	_ = json.NewEncoder(w).Encode(chatToResponsesResponse(chat))
}

func responsesToChatRequest(req responsesRequest) (types.ChatCompletionRequest, error) {
	messages, err := responsesInputMessages(req.Input)
	if err != nil {
		return types.ChatCompletionRequest{}, err
	}
	if req.Instructions != "" {
		messages = append([]types.Message{{
			Role:    "system",
			Content: jsonString(req.Instructions),
		}}, messages...)
	}
	if len(messages) == 0 {
		return types.ChatCompletionRequest{}, fmt.Errorf("input must contain at least one text item")
	}
	return types.ChatCompletionRequest{
		Model:       req.Model,
		Messages:    messages,
		MaxTokens:   req.MaxOutputTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		User:        req.User,
		Stream:      req.Stream,
		Tools:       responsesToolsToChat(req.Tools),
		ToolChoice:  req.ToolChoice,
	}, nil
}

func responsesInputMessages(input any) ([]types.Message, error) {
	switch v := input.(type) {
	case string:
		return []types.Message{{Role: "user", Content: jsonString(v)}}, nil
	case []any:
		out := make([]types.Message, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch stringFromMap(m, "type", "message") {
			case "message":
				role := stringFromMap(m, "role", "user")
				text := responsesContentText(m["content"])
				if text == "" {
					text = responsesContentText(m["text"])
				}
				if strings.TrimSpace(text) != "" {
					out = append(out, types.Message{Role: role, Content: jsonString(text)})
				}
			case "function_call", "custom_tool_call":
				call := types.ToolCall{
					ID:   stringFromMap(m, "call_id", stringFromMap(m, "id", "")),
					Type: "function",
					Function: types.FunctionCall{
						Name:      stringFromMap(m, "name", ""),
						Arguments: stringFromMap(m, "arguments", stringFromMap(m, "input", "{}")),
					},
				}
				if len(out) > 0 && out[len(out)-1].Role == "assistant" {
					out[len(out)-1].ToolCalls = append(out[len(out)-1].ToolCalls, call)
				} else {
					out = append(out, types.Message{Role: "assistant", Content: jsonString(""), ToolCalls: []types.ToolCall{call}})
				}
			case "function_call_output", "custom_tool_call_output":
				out = append(out, types.Message{
					Role:       "tool",
					Content:    jsonString(responsesOutputText(m["output"])),
					ToolCallID: stringFromMap(m, "call_id", ""),
				})
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("input must be a string or array")
	}
}

func responsesToolsToChat(in []responsesTool) []types.Tool {
	out := make([]types.Tool, 0, len(in))
	for _, tool := range in {
		if tool.Type != "function" || tool.Name == "" {
			continue
		}
		out = append(out, types.Tool{
			Type: "function",
			Function: types.FunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			},
		})
	}
	return out
}

func responsesOutputText(value any) string {
	if text := responsesContentText(value); text != "" {
		return text
	}
	b, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return string(b)
}

func responsesContentText(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if t, ok := m["text"].(string); ok {
				b.WriteString(t)
			}
		}
		return b.String()
	default:
		return ""
	}
}

func stringFromMap(m map[string]any, key, fallback string) string {
	if v, ok := m[key].(string); ok && v != "" {
		return v
	}
	return fallback
}

func jsonString(value string) json.RawMessage {
	b, _ := json.Marshal(value)
	return b
}

func chatToResponsesResponse(chat types.ChatCompletionResponse) responsesResponse {
	output := make([]any, 0, len(chat.Choices))
	for i, choice := range chat.Choices {
		if text := choice.Message.ContentString(); text != "" {
			output = append(output, responsesOutput{
				ID:      fmt.Sprintf("msg_%d", i),
				Type:    "message",
				Status:  "completed",
				Role:    "assistant",
				Content: []responsesOutputContent{{Type: "output_text", Text: text}},
			})
		}
		for j, call := range choice.Message.ToolCalls {
			output = append(output, responsesFunctionCall{
				ID:        fmt.Sprintf("fc_%d_%d", i, j),
				Type:      "function_call",
				Status:    "completed",
				CallID:    call.ID,
				Name:      call.Function.Name,
				Arguments: call.Function.Arguments,
			})
		}
	}
	created := chat.Created
	if created == 0 {
		created = time.Now().Unix()
	}
	return responsesResponse{
		ID:        chat.ID,
		Object:    "response",
		CreatedAt: created,
		Model:     chat.Model,
		Output:    output,
		Usage: responsesUsage{
			InputTokens:  chat.Usage.PromptTokens,
			OutputTokens: chat.Usage.CompletionTokens,
			TotalTokens:  chat.Usage.TotalTokens,
		},
	}
}

type responsesStreamWriter struct {
	dst       http.ResponseWriter
	header    http.Header
	status    int
	errorBody bytes.Buffer
	pending   string
	started   bool
	done      bool
	id        string
	model     string
	text      strings.Builder
	toolCalls map[int]*types.ToolCall
	usage     types.Usage
}

func newResponsesStreamWriter(dst http.ResponseWriter) *responsesStreamWriter {
	return &responsesStreamWriter{dst: dst, header: make(http.Header), status: http.StatusOK, toolCalls: map[int]*types.ToolCall{}}
}

func (w *responsesStreamWriter) Header() http.Header { return w.header }

func (w *responsesStreamWriter) WriteHeader(status int) {
	w.status = status
	if status >= http.StatusBadRequest {
		return
	}
	for key, values := range w.header {
		if strings.EqualFold(key, "Content-Type") || strings.EqualFold(key, "Content-Length") {
			continue
		}
		for _, value := range values {
			w.dst.Header().Add(key, value)
		}
	}
	w.dst.Header().Set("Content-Type", "text/event-stream")
	w.dst.Header().Set("Cache-Control", "no-cache")
	w.dst.Header().Set("Connection", "keep-alive")
	w.dst.WriteHeader(status)
}

func (w *responsesStreamWriter) Write(data []byte) (int, error) {
	if w.status >= http.StatusBadRequest {
		return w.errorBody.Write(data)
	}
	w.pending += string(data)
	w.drain()
	return len(data), nil
}

func (w *responsesStreamWriter) Flush() {
	if w.status >= http.StatusBadRequest {
		return
	}
	w.drain()
	if flusher, ok := w.dst.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *responsesStreamWriter) drain() {
	for {
		end := strings.Index(w.pending, "\n\n")
		if end < 0 {
			return
		}
		event := w.pending[:end]
		w.pending = w.pending[end+2:]
		for _, line := range strings.Split(event, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				w.complete()
				continue
			}
			var chunk types.ChatCompletionChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}
			w.consumeChunk(chunk)
		}
	}
}

func (w *responsesStreamWriter) consumeChunk(chunk types.ChatCompletionChunk) {
	if chunk.ID != "" {
		w.id = chunk.ID
	}
	if chunk.Model != "" {
		w.model = chunk.Model
	}
	if !w.started {
		w.started = true
		w.emit(map[string]any{"type": "response.created", "response": map[string]any{"id": w.responseID(), "model": w.model}})
	}
	if chunk.Usage != nil {
		w.usage.MergeFrom(*chunk.Usage)
	}
	for _, choice := range chunk.Choices {
		if choice.Delta.Content != nil {
			w.text.WriteString(*choice.Delta.Content)
		}
		for _, call := range choice.Delta.ToolCalls {
			index := 0
			if call.Index != nil {
				index = *call.Index
			}
			current := w.toolCalls[index]
			if current == nil {
				current = &types.ToolCall{Type: "function"}
				w.toolCalls[index] = current
			}
			if call.ID != "" {
				current.ID = call.ID
			}
			if call.Function.Name != "" {
				current.Function.Name = call.Function.Name
			}
			current.Function.Arguments += call.Function.Arguments
		}
	}
}

func (w *responsesStreamWriter) complete() {
	if w.done {
		return
	}
	w.done = true
	if !w.started {
		w.started = true
		w.emit(map[string]any{"type": "response.created", "response": map[string]any{"id": w.responseID(), "model": w.model}})
	}
	if text := w.text.String(); text != "" {
		w.emit(map[string]any{
			"type": "response.output_item.done",
			"item": map[string]any{
				"id": "msg_" + w.responseID(), "type": "message", "role": "assistant",
				"content": []map[string]any{{"type": "output_text", "text": text}},
			},
		})
	}
	indices := make([]int, 0, len(w.toolCalls))
	for index := range w.toolCalls {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	for _, index := range indices {
		call := w.toolCalls[index]
		w.emit(map[string]any{
			"type": "response.output_item.done",
			"item": map[string]any{
				"id": "fc_" + strconv.Itoa(index), "type": "function_call", "status": "completed",
				"call_id": call.ID, "name": call.Function.Name, "arguments": call.Function.Arguments,
			},
		})
	}
	w.emit(map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id": w.responseID(), "model": w.model,
			"usage": map[string]any{
				"input_tokens": w.usage.PromptTokens, "input_tokens_details": nil,
				"output_tokens": w.usage.CompletionTokens, "output_tokens_details": nil,
				"total_tokens": w.usage.TotalTokens,
			},
		},
	})
}

func (w *responsesStreamWriter) responseID() string {
	if w.id != "" {
		return w.id
	}
	w.id = fmt.Sprintf("resp_%d", time.Now().UnixNano())
	return w.id
}

func (w *responsesStreamWriter) emit(event any) {
	data, err := json.Marshal(event)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w.dst, "data: %s\n\n", data)
}

func (w *responsesStreamWriter) writeCapturedError() {
	if w.status < http.StatusBadRequest {
		return
	}
	for key, values := range w.header {
		for _, value := range values {
			w.dst.Header().Add(key, value)
		}
	}
	w.dst.WriteHeader(w.status)
	_, _ = w.dst.Write(w.errorBody.Bytes())
}

type captureResponseWriter struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func newCaptureResponseWriter() *captureResponseWriter {
	return &captureResponseWriter{header: make(http.Header), status: http.StatusOK}
}

func (w *captureResponseWriter) Header() http.Header { return w.header }

func (w *captureResponseWriter) WriteHeader(status int) { w.status = status }

func (w *captureResponseWriter) Write(b []byte) (int, error) { return w.body.Write(b) }
