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

const (
	vertexDefaultRegion     = "us-east5"
	vertexAnthropicVersion  = "vertex-2023-10-16"
	vertexDefaultMaxTok     = 4096
)

// vertexRequest is the Anthropic Messages API request for Vertex AI.
// Model is conveyed via the URL, not the body.
type vertexRequest struct {
	AnthropicVersion string          `json:"anthropic_version"`
	Messages         []anthropicMsg  `json:"messages"`
	System           string          `json:"system,omitempty"`
	MaxTokens        int             `json:"max_tokens"`
	Stream           bool            `json:"stream,omitempty"`
	Tools            []anthropicTool `json:"tools,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	Stop             json.RawMessage `json:"stop_sequences,omitempty"`
}

// VertexProvider implements Provider for Claude models on Google Vertex AI.
type VertexProvider struct {
	bearerToken string
	projectID   string
	region      string
	baseURL     string
	client      *http.Client
}

// NewVertex creates a new Vertex AI provider from the given configuration.
func NewVertex(cfg *config.ProviderConfig) *VertexProvider {
	region := vertexDefaultRegion
	if cfg.Region != "" {
		region = cfg.Region
	}

	base := fmt.Sprintf("https://%s-aiplatform.googleapis.com", region)
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}

	return &VertexProvider{
		bearerToken: cfg.APIKey,
		projectID:   cfg.ProjectID,
		region:      region,
		baseURL:     base,
		client:      &http.Client{},
	}
}

// Name returns "vertex".
func (p *VertexProvider) Name() string { return "vertex" }

// Send sends a non-streaming request to Vertex AI's rawPredict endpoint.
func (p *VertexProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	vReq := p.translateRequest(req, false)

	body, err := json.Marshal(vReq)
	if err != nil {
		return nil, fmt.Errorf("vertex: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:rawPredict",
		p.baseURL, p.projectID, p.region, model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("vertex: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("vertex: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vertex: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var aResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&aResp); err != nil {
		return nil, fmt.Errorf("vertex: decode response: %w", err)
	}

	// Vertex may not return model in the body; fill it in.
	if aResp.Model == "" {
		aResp.Model = model
	}

	chatResp := translateAnthropicResponse(&aResp)
	return &Response{
		ChatResponse: chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming request to Vertex AI's streamRawPredict endpoint.
func (p *VertexProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	vReq := p.translateRequest(req, true)

	body, err := json.Marshal(vReq)
	if err != nil {
		return nil, fmt.Errorf("vertex: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:streamRawPredict",
		p.baseURL, p.projectID, p.region, model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("vertex: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("vertex: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("vertex: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &anthropicStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

func (p *VertexProvider) translateRequest(req *types.ChatCompletionRequest, stream bool) *vertexRequest {
	vReq := &vertexRequest{
		AnthropicVersion: vertexAnthropicVersion,
		MaxTokens:        vertexDefaultMaxTok,
		Stream:           stream,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		Stop:             req.Stop,
	}

	if req.MaxTokens != nil {
		vReq.MaxTokens = *req.MaxTokens
	}

	for _, m := range req.Messages {
		if m.Role == "system" {
			vReq.System = m.ContentString()
			continue
		}
		vReq.Messages = append(vReq.Messages, anthropicMsg{
			Role:    m.Role,
			Content: m.ContentString(),
		})
	}

	for _, t := range req.Tools {
		vReq.Tools = append(vReq.Tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	return vReq
}

func (p *VertexProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.bearerToken)
}
