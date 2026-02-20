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

const geminiDefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta/openai"

// GeminiProvider implements Provider for Google's Gemini API via its
// OpenAI-compatible endpoint.
type GeminiProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewGemini creates a new Gemini provider from the given configuration.
func NewGemini(cfg *config.ProviderConfig) *GeminiProvider {
	base := geminiDefaultBaseURL
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}
	return &GeminiProvider{
		apiKey:  cfg.APIKey,
		baseURL: base,
		client:  &http.Client{},
	}
}

// Name returns "gemini".
func (p *GeminiProvider) Name() string { return "gemini" }

// Send sends a non-streaming chat completion request to Gemini.
func (p *GeminiProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	payload := *req
	payload.Model = model
	payload.Stream = false

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp types.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}

	return &Response{
		ChatResponse: &chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming chat completion request to Gemini and returns a StreamReader.
func (p *GeminiProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	payload := *req
	payload.Model = model
	payload.Stream = true

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("gemini: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &sseStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

func (p *GeminiProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}
