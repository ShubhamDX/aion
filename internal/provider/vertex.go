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
	vertexDefaultRegion    = "us-east5"
	vertexAnthropicVersion = "vertex-2023-10-16"
	vertexDefaultMaxTok    = 4096
)

// vertexRequest is the Anthropic Messages API request for Vertex AI.
// Model is conveyed via the URL, not the body.
type vertexRequest struct {
	AnthropicVersion string          `json:"anthropic_version"`
	Messages         []anthropicMsg  `json:"messages"`
	System           any             `json:"system,omitempty"`
	MaxTokens        int             `json:"max_tokens"`
	Stream           bool            `json:"stream,omitempty"`
	Tools            []anthropicTool `json:"tools,omitempty"`
	ToolChoice       json.RawMessage `json:"tool_choice,omitempty"`
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
	choice, err := vertexToolChoice(req)
	if err != nil {
		return nil, err
	}
	vReq := p.translateRequest(req, false)
	vReq.ToolChoice = choice

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
	choice, err := vertexToolChoice(req)
	if err != nil {
		return nil, err
	}
	vReq := p.translateRequest(req, true)
	vReq.ToolChoice = choice

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

	vReq.System, vReq.Messages = translateAnthropicMessages(req.Messages)
	vReq.System, vReq.Messages = translateClaudeCacheMessages(req.Messages, vReq.System)

	for _, t := range req.Tools {
		vReq.Tools = append(vReq.Tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	return vReq
}

func vertexToolChoice(req *types.ChatCompletionRequest) (json.RawMessage, error) {
	if req.ResponseFormat != nil && req.ResponseFormat.Type != "" && req.ResponseFormat.Type != "text" {
		return nil, fmt.Errorf("vertex: response_format is not supported on this model transport")
	}
	if req.SchemaSettings != nil && req.SchemaSettings.MustEmitNative {
		return nil, fmt.Errorf("vertex: mandatory native schema is not supported on this model transport")
	}
	return claudeToolChoice(req, "vertex")
}

func (p *VertexProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.bearerToken)
}
