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

	"github.com/ShubhamDX/aion/internal/types"
)

func isBedrockMantleModel(model string) bool {
	return strings.HasPrefix(model, "openai.gpt-5.6-")
}

func validReasoningEffort(effort string) bool {
	switch effort {
	case "none", "low", "medium", "high", "xhigh", "max":
		return true
	}
	return false
}

// Mantle uses OpenAI chat JSON and SSE, not Anthropic invoke event streams.
func (p *BedrockProvider) mantleRequest(ctx context.Context, req *types.ChatCompletionRequest, model string, stream bool) (*http.Response, bool, error) {
	payload, native := openAINativePayload(req, model, stream)
	if payload.ReasoningEffort == "" {
		payload.ReasoningEffort = p.reasoningEfforts[model]
	}
	if payload.ReasoningEffort != "" && !validReasoningEffort(payload.ReasoningEffort) {
		return nil, false, fmt.Errorf("bedrock mantle: invalid reasoning_effort")
	}
	maxTokens := payload.MaxTokens
	payload.MaxTokens = nil
	body, err := json.Marshal(struct {
		types.ChatCompletionRequest
		MaxCompletionTokens *int `json:"max_completion_tokens,omitempty"`
		StreamOptions       any  `json:"stream_options,omitempty"`
	}{payload, maxTokens, mantleStreamOptions(stream)})
	if err != nil {
		return nil, false, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.mantleBaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	if err = p.authorize(ctx, request, body); err != nil {
		return nil, false, err
	}
	response, err := p.client.Do(request)
	if err != nil {
		return nil, false, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer response.Body.Close()
		// Provider errors can contain echoed input. Keep customer data out of errors.
		io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
		return nil, false, fmt.Errorf("bedrock mantle: HTTP %d", response.StatusCode)
	}
	return response, native, nil
}

func mantleStreamOptions(stream bool) any {
	if stream {
		return map[string]bool{"include_usage": true}
	}
	return nil
}

func (p *BedrockProvider) sendMantle(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error) {
	response, native, err := p.mantleRequest(ctx, req, model, false)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	var chat types.ChatCompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&chat); err != nil {
		return nil, err
	}
	chat.Usage.NormalizeInputPartition()
	return &Response{ChatResponse: &chat, StatusCode: response.StatusCode, SchemaNativeEmitted: native}, nil
}

func (p *BedrockProvider) streamMantle(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error) {
	response, _, err := p.mantleRequest(ctx, req, model, true)
	if err != nil {
		return nil, err
	}
	return &sseStreamReader{reader: bufio.NewReader(response.Body), body: response.Body}, nil
}
