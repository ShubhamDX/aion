package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

// LocalProvider implements Provider for a local llama.cpp server
// (llama-server with an OpenAI-compatible /v1/chat/completions endpoint).
type LocalProvider struct {
	baseURL string
	client  *http.Client
	process *LlamaProcess
}

// NewLocal creates a new local llama.cpp provider.
func NewLocal(cfg *config.LocalProviderConfig) *LocalProvider {
	base := strings.TrimRight(cfg.BaseURL, "/")
	return &LocalProvider{
		baseURL: base,
		client: &http.Client{
			Timeout: 300 * time.Second,
		},
	}
}

// Name returns "local".
func (p *LocalProvider) Name() string { return "local" }

// Send sends a non-streaming chat completion request to the local llama-server.
func (p *LocalProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	payload := upstreamPayload(req, model, false)

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("local: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("local: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("local: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("local: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp types.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("local: decode response: %w", err)
	}

	return &Response{
		ChatResponse: &chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming chat completion request to the local llama-server
// and returns a StreamReader. Reuses sseStreamReader since llama-server uses
// the same SSE format as OpenAI.
func (p *LocalProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	payload := upstreamPayload(req, model, true)

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("local: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("local: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("local: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("local: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &sseStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

// SetProcess assigns a managed LlamaProcess to this provider.
func (p *LocalProvider) SetProcess(proc *LlamaProcess) {
	p.process = proc
}

// WaitReady polls the llama-server /health endpoint until it returns 200
// or the context/timeout expires. llama-server returns 503 while loading.
func (p *LocalProvider) WaitReady(ctx context.Context, timeout time.Duration) error {
	healthURL := p.baseURL + "/health"
	// Strip /v1 suffix if present for the health endpoint, which is at the root.
	healthURL = strings.Replace(healthURL, "/v1/health", "/health", 1)

	deadline := time.After(timeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	slog.Info("waiting for local llama-server", "url", healthURL, "timeout", timeout)

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("local: context cancelled while waiting for server: %w", ctx.Err())
		case <-deadline:
			return fmt.Errorf("local: server not ready within %s", timeout)
		case <-ticker.C:
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
			if err != nil {
				continue
			}
			resp, err := p.client.Do(req)
			if err != nil {
				continue
			}
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				slog.Info("local llama-server is ready")
				return nil
			}
		}
	}
}

// Shutdown stops the managed llama-server process if one is running.
func (p *LocalProvider) Shutdown() error {
	if p.process != nil {
		slog.Info("stopping managed llama-server")
		return p.process.Stop()
	}
	return nil
}
