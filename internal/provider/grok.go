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

const grokDefaultBaseURL = "https://api.x.ai/v1"

// GrokProvider implements Provider for xAI's Grok API via its
// OpenAI-compatible endpoint.
type GrokProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewGrok creates a new Grok provider from the given configuration.
func NewGrok(cfg *config.ProviderConfig) *GrokProvider {
	base := grokDefaultBaseURL
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}
	return &GrokProvider{
		apiKey:  cfg.APIKey,
		baseURL: base,
		client:  &http.Client{},
	}
}

// Name returns "grok".
func (p *GrokProvider) Name() string { return "grok" }

// Send sends a non-streaming chat completion request to Grok.
func (p *GrokProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	payload := upstreamPayload(req, model, false)

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("grok: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("grok: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("grok: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("grok: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp types.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("grok: decode response: %w", err)
	}

	return &Response{
		ChatResponse: &chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming chat completion request to Grok and returns a StreamReader.
func (p *GrokProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	payload := upstreamPayload(req, model, true)

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("grok: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("grok: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("grok: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("grok: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &sseStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

func (p *GrokProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}
