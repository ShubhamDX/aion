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
	bedrockDefaultRegion     = "us-east-1"
	bedrockAnthropicVersion  = "bedrock-2023-05-31"
	bedrockDefaultMaxTok     = 4096
)

// bedrockRequest is the Anthropic Messages API request for Bedrock.
// Model is conveyed via the URL, not the body. AnthropicVersion is a
// required body field instead of a header.
type bedrockRequest struct {
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

// BedrockProvider implements Provider for Claude models on AWS Bedrock.
type BedrockProvider struct {
	bearerToken string
	region      string
	baseURL     string
	client      *http.Client
}

// NewBedrock creates a new Bedrock provider from the given configuration.
func NewBedrock(cfg *config.ProviderConfig) *BedrockProvider {
	region := bedrockDefaultRegion
	if cfg.Region != "" {
		region = cfg.Region
	}

	base := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", region)
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}

	return &BedrockProvider{
		bearerToken: cfg.APIKey,
		region:      region,
		baseURL:     base,
		client:      &http.Client{},
	}
}

// Name returns "bedrock".
func (p *BedrockProvider) Name() string { return "bedrock" }

// Send sends a non-streaming request to Bedrock's invoke endpoint.
func (p *BedrockProvider) Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	bReq := p.translateRequest(req, false)

	body, err := json.Marshal(bReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/model/%s/invoke", p.baseURL, model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var aResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&aResp); err != nil {
		return nil, fmt.Errorf("bedrock: decode response: %w", err)
	}

	// Bedrock doesn't return model in the body; fill it in.
	if aResp.Model == "" {
		aResp.Model = model
	}

	chatResp := translateAnthropicResponse(&aResp)
	return &Response{
		ChatResponse: chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// SendStream sends a streaming request to Bedrock's invoke-with-response-stream endpoint.
func (p *BedrockProvider) SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	bReq := p.translateRequest(req, true)

	body, err := json.Marshal(bReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/model/%s/invoke-with-response-stream", p.baseURL, model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: create request: %w", err)
	}
	p.setHeaders(httpReq)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("bedrock: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &anthropicStreamReader{
		reader: bufio.NewReader(resp.Body),
		body:   resp.Body,
	}, nil
}

func (p *BedrockProvider) translateRequest(req *types.ChatCompletionRequest, stream bool) *bedrockRequest {
	bReq := &bedrockRequest{
		AnthropicVersion: bedrockAnthropicVersion,
		MaxTokens:        bedrockDefaultMaxTok,
		Stream:           stream,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		Stop:             req.Stop,
	}

	if req.MaxTokens != nil {
		bReq.MaxTokens = *req.MaxTokens
	}

	for _, m := range req.Messages {
		if m.Role == "system" {
			bReq.System = m.ContentString()
			continue
		}
		bReq.Messages = append(bReq.Messages, anthropicMsg{
			Role:    m.Role,
			Content: m.ContentString(),
		})
	}

	for _, t := range req.Tools {
		bReq.Tools = append(bReq.Tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	return bReq
}

func (p *BedrockProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.bearerToken)
}
