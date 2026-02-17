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

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

const openAIDefaultBaseURL = "https://api.openai.com/v1"

// OpenAIProvider implements Provider for the OpenAI API.
type OpenAIProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewOpenAI creates a new OpenAI provider from the given configuration.
func NewOpenAI(cfg *config.ProviderConfig) *OpenAIProvider {
	base := openAIDefaultBaseURL
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}
	return &OpenAIProvider{
		apiKey:  cfg.APIKey,
		baseURL: base,
		client:  &http.Client{},
	}
}

// Name returns "openai".
func (p *OpenAIProvider) Name() string { return "openai" }

// Send sends a non-streaming chat completion request to OpenAI.
func (p *OpenAIProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	payload := *req
	payload.Model = model
	payload.Stream = false

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp types.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	return &Response{
		ChatResponse: &chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming chat completion request to OpenAI and returns a StreamReader.
func (p *OpenAIProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	payload := *req
	payload.Model = model
	payload.Stream = true

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &sseStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

func (p *OpenAIProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}

// sseStreamReader reads SSE lines from an HTTP response body and parses
// them into ChatCompletionChunk values.
type sseStreamReader struct {
	reader *bufio.Reader
	body   io.ReadCloser
}

// ReadChunk reads the next chunk from the SSE stream.
// Returns io.EOF when the stream sends "data: [DONE]".
func (s *sseStreamReader) ReadChunk() (*types.ChatCompletionChunk, error) {
	for {
		line, err := s.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil, io.EOF
			}
			return nil, fmt.Errorf("sse: read line: %w", err)
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return nil, io.EOF
		}

		var chunk types.ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("sse: unmarshal chunk: %w", err)
		}

		return &chunk, nil
	}
}

// Close closes the underlying response body.
func (s *sseStreamReader) Close() error {
	return s.body.Close()
}
