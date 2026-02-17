package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

const (
	anthropicDefaultBaseURL = "https://api.anthropic.com/v1"
	anthropicVersion        = "2023-06-01"
	anthropicDefaultMaxTok  = 4096
)

// ---------- internal Anthropic API types ----------

type anthropicRequest struct {
	Model       string          `json:"model"`
	Messages    []anthropicMsg  `json:"messages"`
	System      string          `json:"system,omitempty"`
	MaxTokens   int             `json:"max_tokens"`
	Stream      bool            `json:"stream,omitempty"`
	Tools       []anthropicTool `json:"tools,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stop        json.RawMessage `json:"stop_sequences,omitempty"`
}

type anthropicMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicResponse struct {
	ID         string             `json:"id"`
	Type       string             `json:"type"`
	Role       string             `json:"role"`
	Content    []anthropicContent `json:"content"`
	Model      string             `json:"model"`
	StopReason string             `json:"stop_reason"`
	Usage      anthropicUsage     `json:"usage"`
}

type anthropicContent struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ---------- streaming event types ----------

type anthropicStreamEvent struct {
	Type  string          `json:"type"`
	Delta json.RawMessage `json:"delta,omitempty"`
	Index int             `json:"index,omitempty"`

	// message_start carries the full message stub
	Message *anthropicResponse `json:"message,omitempty"`

	// content_block_start
	ContentBlock *anthropicContent `json:"content_block,omitempty"`
}

type anthropicDelta struct {
	Type       string `json:"type,omitempty"`
	Text       string `json:"text,omitempty"`
	StopReason string `json:"stop_reason,omitempty"`
}

// ---------- AnthropicProvider ----------

// AnthropicProvider implements Provider for the Anthropic Messages API.
type AnthropicProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewAnthropic creates a new Anthropic provider from the given configuration.
func NewAnthropic(cfg *config.ProviderConfig) *AnthropicProvider {
	base := anthropicDefaultBaseURL
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}
	return &AnthropicProvider{
		apiKey:  cfg.APIKey,
		baseURL: base,
		client:  &http.Client{},
	}
}

// Name returns "anthropic".
func (p *AnthropicProvider) Name() string { return "anthropic" }

// Send sends a non-streaming request translated to the Anthropic Messages API.
func (p *AnthropicProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	aReq := p.translateRequest(req, model, false)

	body, err := json.Marshal(aReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var aResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&aResp); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}

	chatResp := p.translateResponse(&aResp)
	return &Response{
		ChatResponse: chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming request to the Anthropic Messages API.
func (p *AnthropicProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	aReq := p.translateRequest(req, model, true)

	body, err := json.Marshal(aReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("anthropic: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &anthropicStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

// ---------- request translation ----------

func (p *AnthropicProvider) translateRequest(req *types.ChatCompletionRequest, model string, stream bool) *anthropicRequest {
	aReq := &anthropicRequest{
		Model:       model,
		MaxTokens:   anthropicDefaultMaxTok,
		Stream:      stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.Stop,
	}

	if req.MaxTokens != nil {
		aReq.MaxTokens = *req.MaxTokens
	}

	// Extract system message and convert the rest.
	for _, m := range req.Messages {
		if m.Role == "system" {
			aReq.System = m.ContentString()
			continue
		}
		aReq.Messages = append(aReq.Messages, anthropicMsg{
			Role:    m.Role,
			Content: m.ContentString(),
		})
	}

	// Convert tools.
	for _, t := range req.Tools {
		aReq.Tools = append(aReq.Tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	return aReq
}

// ---------- response translation ----------

func (p *AnthropicProvider) translateResponse(aResp *anthropicResponse) *types.ChatCompletionResponse {
	msg := types.Message{
		Role: "assistant",
	}

	var toolCalls []types.ToolCall
	for _, block := range aResp.Content {
		switch block.Type {
		case "text":
			b, _ := json.Marshal(block.Text)
			msg.Content = b
		case "tool_use":
			args := string(block.Input)
			toolCalls = append(toolCalls, types.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: types.FunctionCall{
					Name:      block.Name,
					Arguments: args,
				},
			})
		}
	}
	msg.ToolCalls = toolCalls

	finishReason := mapAnthropicStopReason(aResp.StopReason)

	return &types.ChatCompletionResponse{
		ID:      aResp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   aResp.Model,
		Choices: []types.Choice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: finishReason,
			},
		},
		Usage: types.Usage{
			PromptTokens:     aResp.Usage.InputTokens,
			CompletionTokens: aResp.Usage.OutputTokens,
			TotalTokens:      aResp.Usage.InputTokens + aResp.Usage.OutputTokens,
		},
	}
}

func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	default:
		return reason
	}
}

func (p *AnthropicProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)
}

// ---------- streaming ----------

type anthropicStreamReader struct {
	reader *bufio.Reader
	body   io.ReadCloser
	id     string
	model  string
}

// ReadChunk reads the next SSE event from the Anthropic stream and translates
// it to an OpenAI-compatible ChatCompletionChunk.
func (s *anthropicStreamReader) ReadChunk() (*types.ChatCompletionChunk, error) {
	for {
		eventType, data, err := s.readSSEEvent()
		if err != nil {
			return nil, err
		}

		switch eventType {
		case "message_start":
			var evt anthropicStreamEvent
			if err := json.Unmarshal(data, &evt); err != nil {
				return nil, fmt.Errorf("anthropic stream: unmarshal message_start: %w", err)
			}
			if evt.Message != nil {
				s.id = evt.Message.ID
				s.model = evt.Message.Model
			}
			// Emit a chunk with the assistant role.
			role := "assistant"
			return &types.ChatCompletionChunk{
				ID:      s.id,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   s.model,
				Choices: []types.ChunkChoice{
					{
						Index: 0,
						Delta: types.ChunkDelta{
							Role: role,
						},
					},
				},
			}, nil

		case "content_block_delta":
			var evt anthropicStreamEvent
			if err := json.Unmarshal(data, &evt); err != nil {
				return nil, fmt.Errorf("anthropic stream: unmarshal content_block_delta: %w", err)
			}
			var delta anthropicDelta
			if err := json.Unmarshal(evt.Delta, &delta); err != nil {
				return nil, fmt.Errorf("anthropic stream: unmarshal delta: %w", err)
			}
			content := delta.Text
			return &types.ChatCompletionChunk{
				ID:      s.id,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   s.model,
				Choices: []types.ChunkChoice{
					{
						Index: 0,
						Delta: types.ChunkDelta{
							Content: &content,
						},
					},
				},
			}, nil

		case "message_delta":
			var evt anthropicStreamEvent
			if err := json.Unmarshal(data, &evt); err != nil {
				return nil, fmt.Errorf("anthropic stream: unmarshal message_delta: %w", err)
			}
			var delta anthropicDelta
			if err := json.Unmarshal(evt.Delta, &delta); err != nil {
				return nil, fmt.Errorf("anthropic stream: unmarshal delta: %w", err)
			}
			finishReason := mapAnthropicStopReason(delta.StopReason)
			return &types.ChatCompletionChunk{
				ID:      s.id,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   s.model,
				Choices: []types.ChunkChoice{
					{
						Index:        0,
						Delta:        types.ChunkDelta{},
						FinishReason: &finishReason,
					},
				},
			}, nil

		case "message_stop":
			return nil, io.EOF

		default:
			// Skip unknown event types (ping, content_block_start, content_block_stop, etc.)
			continue
		}
	}
}

// readSSEEvent reads lines until it has a complete SSE event (event + data).
func (s *anthropicStreamReader) readSSEEvent() (eventType string, data []byte, err error) {
	for {
		line, readErr := s.reader.ReadString('\n')
		if readErr != nil {
			if readErr == io.EOF {
				return "", nil, io.EOF
			}
			return "", nil, fmt.Errorf("anthropic stream: read line: %w", readErr)
		}

		line = strings.TrimSpace(line)
		if line == "" {
			// Empty line = end of event. If we have data, return it.
			if len(data) > 0 {
				return eventType, data, nil
			}
			continue
		}

		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data = []byte(strings.TrimPrefix(line, "data: "))
			continue
		}
	}
}

// Close closes the underlying response body.
func (s *anthropicStreamReader) Close() error {
	return s.body.Close()
}
