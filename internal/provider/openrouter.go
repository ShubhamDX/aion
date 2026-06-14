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

const openRouterDefaultBaseURL = "https://openrouter.ai/api/v1"

// OpenRouterProvider implements Provider for the OpenRouter API.
// OpenRouter exposes an OpenAI-compatible interface with additional headers.
type OpenRouterProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewOpenRouter creates a new OpenRouter provider from the given configuration.
func NewOpenRouter(cfg *config.ProviderConfig) *OpenRouterProvider {
	base := openRouterDefaultBaseURL
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}
	return &OpenRouterProvider{
		apiKey:  cfg.APIKey,
		baseURL: base,
		client:  &http.Client{},
	}
}

// Name returns "openrouter".
func (p *OpenRouterProvider) Name() string { return "openrouter" }

// Send sends a non-streaming chat completion request to OpenRouter.
func (p *OpenRouterProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	payload := upstreamPayload(req, model, false)

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openrouter: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openrouter: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openrouter: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp types.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("openrouter: decode response: %w", err)
	}

	return &Response{
		ChatResponse: &chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming chat completion request to OpenRouter and returns a StreamReader.
func (p *OpenRouterProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	payload := upstreamPayload(req, model, true)

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openrouter: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openrouter: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openrouter: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &sseStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

func (p *OpenRouterProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("HTTP-Referer", "https://github.com/ShubhamDX/aion")
	req.Header.Set("X-Title", "AION Gateway")
}
