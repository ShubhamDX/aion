package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ShubhamDX/aion/internal/types"
)

type responsesRequest struct {
	Model           string   `json:"model"`
	Input           any      `json:"input"`
	Instructions    string   `json:"instructions,omitempty"`
	MaxOutputTokens *int     `json:"max_output_tokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	Stream          bool     `json:"stream,omitempty"`
	User            string   `json:"user,omitempty"`
}

type responsesResponse struct {
	ID        string                 `json:"id"`
	Object    string                 `json:"object"`
	CreatedAt int64                  `json:"created_at"`
	Model     string                 `json:"model"`
	Output    []responsesOutput      `json:"output"`
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
	if req.Stream {
		writeError(w, http.StatusBadRequest, "unsupported_request", "streaming /v1/responses is not supported")
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

	rec := newCaptureResponseWriter()
	r2 := r.Clone(r.Context())
	r2.Body = io.NopCloser(bytes.NewReader(body))
	r2.ContentLength = int64(len(body))
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
			role := stringFromMap(m, "role", "user")
			text := responsesContentText(m["content"])
			if text == "" {
				text = responsesContentText(m["text"])
			}
			if strings.TrimSpace(text) == "" {
				continue
			}
			out = append(out, types.Message{Role: role, Content: jsonString(text)})
		}
		return out, nil
	default:
		return nil, fmt.Errorf("input must be a string or array")
	}
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
	output := make([]responsesOutput, 0, len(chat.Choices))
	for i, choice := range chat.Choices {
		output = append(output, responsesOutput{
			ID:     fmt.Sprintf("msg_%d", i),
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []responsesOutputContent{{
				Type: "output_text",
				Text: choice.Message.ContentString(),
			}},
		})
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
