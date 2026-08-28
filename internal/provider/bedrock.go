package provider

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials/stscreds"
	"github.com/aws/aws-sdk-go-v2/service/sts"
)

const (
	bedrockDefaultRegion    = "us-east-1"
	bedrockAnthropicVersion = "bedrock-2023-05-31"
	bedrockDefaultMaxTok    = 4096
)

// bedrockRequest is the Anthropic Messages API request for Bedrock.
// Model is conveyed via the URL, not the body. AnthropicVersion is a
// required body field instead of a header.
type bedrockRequest struct {
	AnthropicVersion string          `json:"anthropic_version"`
	Messages         []anthropicMsg  `json:"messages"`
	System           string          `json:"system,omitempty"`
	MaxTokens        int             `json:"max_tokens"`
	Stream           bool            `json:"-"`
	Tools            []anthropicTool `json:"tools,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	Stop             json.RawMessage `json:"stop_sequences,omitempty"`
}

// BedrockProvider implements Provider for Claude models on AWS Bedrock.
type BedrockProvider struct {
	bearerToken string
	credentials aws.CredentialsProvider
	signer      *v4.Signer
	region      string
	baseURL     string
	client      *http.Client
}

// NewBedrock creates a new Bedrock provider from the given configuration.
func NewBedrock(cfg *config.ProviderConfig) (*BedrockProvider, error) {
	region := bedrockDefaultRegion
	if cfg.Region != "" {
		region = cfg.Region
	}

	base := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", region)
	if cfg.BaseURL != "" {
		base = strings.TrimRight(cfg.BaseURL, "/")
	}

	provider := &BedrockProvider{
		region:  region,
		baseURL: base,
		client:  &http.Client{},
	}
	mode := cfg.CredentialMode
	if mode == "" {
		if cfg.APIKey != "" {
			mode = "bearer"
		} else {
			mode = "aws_sdk"
		}
	}
	if mode == "bearer" {
		if cfg.APIKey == "" {
			return nil, fmt.Errorf("bearer credential mode requires api_key")
		}
		provider.bearerToken = cfg.APIKey
		return provider, nil
	}

	awsCfg, err := awsconfig.LoadDefaultConfig(context.Background(), awsconfig.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("load AWS credentials: %w", err)
	}
	credentials := awsCfg.Credentials
	if mode == "assume_role" {
		if cfg.RoleARN == "" {
			return nil, fmt.Errorf("assume_role credential mode requires role_arn")
		}
		assume := stscreds.NewAssumeRoleProvider(sts.NewFromConfig(awsCfg), cfg.RoleARN, func(options *stscreds.AssumeRoleOptions) {
			if cfg.ExternalID != "" {
				options.ExternalID = aws.String(cfg.ExternalID)
			}
			if cfg.SessionName != "" {
				options.RoleSessionName = cfg.SessionName
			}
		})
		credentials = aws.NewCredentialsCache(assume)
	}
	provider.credentials = credentials
	provider.signer = v4.NewSigner()
	return provider, nil
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
	if err := p.authorize(ctx, httpReq, body); err != nil {
		return nil, err
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("bedrock: read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("bedrock: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	chatResp, err := parseBedrockResponse(respBody, model)
	if err != nil {
		return nil, fmt.Errorf("bedrock: model %q: %w", model, err)
	}

	return &Response{
		ChatResponse: chatResp,
		StatusCode:   resp.StatusCode,
	}, nil
}

// parseBedrockResponse decodes a Bedrock invoke response body into an
// OpenAI-style ChatCompletionResponse. Bedrock's invoke endpoint returns a
// different envelope per model family: Anthropic Claude models return the
// Messages API shape (a top-level "content" block array and "usage" with
// input_tokens/output_tokens), while other model families hosted on
// Bedrock, e.g. Qwen3, return an OpenAI-compatible shape directly
// (top-level "choices" and "usage" with prompt_tokens/completion_tokens).
// Decoding the wrong shape into anthropicResponse doesn't fail: unknown
// fields are silently ignored, leaving content nil and usage all zero. So
// the envelope is detected by its distinguishing top-level field before
// decoding, instead of assuming one shape for every model.
func parseBedrockResponse(body []byte, model string) (*types.ChatCompletionResponse, error) {
	var probe struct {
		Content json.RawMessage `json:"content"`
		Choices json.RawMessage `json:"choices"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	switch {
	case probe.Choices != nil:
		var chatResp types.ChatCompletionResponse
		if err := json.Unmarshal(body, &chatResp); err != nil {
			return nil, fmt.Errorf("decode OpenAI-style response: %w", err)
		}
		if chatResp.Model == "" {
			chatResp.Model = model
		}
		return &chatResp, nil
	case probe.Content != nil:
		var aResp anthropicResponse
		if err := json.Unmarshal(body, &aResp); err != nil {
			return nil, fmt.Errorf("decode Anthropic-style response: %w", err)
		}
		if aResp.Model == "" {
			aResp.Model = model
		}
		return translateAnthropicResponse(&aResp), nil
	default:
		return nil, fmt.Errorf("unrecognized response shape (no top-level \"content\" or \"choices\" field): %s", string(body))
	}
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
	if err := p.authorize(ctx, httpReq, body); err != nil {
		return nil, err
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("bedrock: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return &bedrockStreamReader{
		reader: resp.Body,
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

	bReq.System, bReq.Messages = translateAnthropicMessages(req.Messages)

	for _, t := range req.Tools {
		bReq.Tools = append(bReq.Tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	return bReq
}

func (p *BedrockProvider) authorize(ctx context.Context, req *http.Request, body []byte) error {
	req.Header.Set("Content-Type", "application/json")
	if p.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+p.bearerToken)
		return nil
	}
	if p.credentials == nil || p.signer == nil {
		return fmt.Errorf("bedrock: AWS credentials are not configured")
	}
	credentials, err := p.credentials.Retrieve(ctx)
	if err != nil {
		return fmt.Errorf("bedrock: retrieve AWS credentials: %w", err)
	}
	sum := sha256.Sum256(body)
	payloadHash := hex.EncodeToString(sum[:])
	if err := p.signer.SignHTTP(ctx, credentials, req, payloadHash, "bedrock", p.region, time.Now().UTC()); err != nil {
		return fmt.Errorf("bedrock: sign request: %w", err)
	}
	return nil
}

// ---------- Bedrock streaming (AWS Event Stream binary format) ----------

// bedrockEventPayload wraps the base64-encoded Anthropic event in each Bedrock chunk.
type bedrockEventPayload struct {
	Bytes string `json:"bytes"`
}

// bedrockStreamReader reads AWS Event Stream binary frames from Bedrock's
// invoke-with-response-stream endpoint and translates them to OpenAI-compatible chunks.
type bedrockStreamReader struct {
	reader     io.Reader
	body       io.ReadCloser
	id         string
	model      string
	toolBlocks map[int]types.ToolCall
}

// ReadChunk reads the next event from the Bedrock binary stream and translates
// it to an OpenAI-compatible ChatCompletionChunk.
func (s *bedrockStreamReader) ReadChunk() (*types.ChatCompletionChunk, error) {
	for {
		headers, payload, err := s.readFrame()
		if err != nil {
			if err == io.EOF {
				return nil, io.EOF
			}
			return nil, fmt.Errorf("bedrock stream: %w", err)
		}

		msgType := headers[":message-type"]
		if msgType == "exception" {
			exType := headers[":exception-type"]
			return nil, fmt.Errorf("bedrock stream: exception %s: %s", exType, string(payload))
		}

		if msgType != "event" {
			continue
		}

		if headers[":event-type"] != "chunk" {
			continue
		}

		// Parse {"bytes":"..."} wrapper.
		var ep bedrockEventPayload
		if err := json.Unmarshal(payload, &ep); err != nil {
			return nil, fmt.Errorf("bedrock stream: unmarshal payload: %w", err)
		}

		decoded, err := base64.StdEncoding.DecodeString(ep.Bytes)
		if err != nil {
			return nil, fmt.Errorf("bedrock stream: base64 decode: %w", err)
		}

		// Parse the Anthropic streaming event.
		var evt anthropicStreamEvent
		if err := json.Unmarshal(decoded, &evt); err != nil {
			return nil, fmt.Errorf("bedrock stream: unmarshal event: %w", err)
		}

		chunk, done := s.translateEvent(&evt)
		if done {
			return nil, io.EOF
		}
		if chunk != nil {
			return chunk, nil
		}
	}
}

func (s *bedrockStreamReader) translateEvent(evt *anthropicStreamEvent) (chunk *types.ChatCompletionChunk, done bool) {
	switch evt.Type {
	case "message_start":
		if evt.Message != nil {
			s.id = evt.Message.ID
			s.model = evt.Message.Model
		}
		var usage *types.Usage
		if evt.Message != nil {
			u := usageFromAnthropic(evt.Message.Usage)
			usage = &u
		} else if evt.Usage != nil {
			u := usageFromAnthropic(*evt.Usage)
			usage = &u
		}
		role := "assistant"
		return &types.ChatCompletionChunk{
			ID:      s.id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   s.model,
			Choices: []types.ChunkChoice{
				{
					Index: 0,
					Delta: types.ChunkDelta{Role: role},
				},
			},
			Usage: usage,
		}, false

	case "content_block_delta":
		chunk, err := translateAnthropicContentBlockDelta(s.id, s.model, s.toolBlocks, evt)
		if err != nil {
			return nil, false
		}
		return chunk, false

	case "content_block_start":
		var chunk *types.ChatCompletionChunk
		chunk, s.toolBlocks = translateAnthropicContentBlockStart(s.id, s.model, s.toolBlocks, evt)
		return chunk, false

	case "message_delta":
		var delta anthropicDelta
		if err := json.Unmarshal(evt.Delta, &delta); err != nil {
			return nil, false
		}
		finishReason := mapAnthropicStopReason(delta.StopReason)
		var usage *types.Usage
		if evt.Usage != nil {
			u := usageFromAnthropic(*evt.Usage)
			usage = &u
		}
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
			Usage: usage,
		}, false

	case "message_stop":
		return nil, true

	default:
		return nil, false
	}
}

// readFrame reads a single AWS Event Stream binary frame.
// Frame layout: [total_len:4][headers_len:4][prelude_crc:4][headers:*][payload:*][msg_crc:4]
func (s *bedrockStreamReader) readFrame() (headers map[string]string, payload []byte, err error) {
	// Read 12-byte prelude.
	prelude := make([]byte, 12)
	if _, err := io.ReadFull(s.reader, prelude); err != nil {
		return nil, nil, err
	}

	totalLen := binary.BigEndian.Uint32(prelude[0:4])
	headersLen := binary.BigEndian.Uint32(prelude[4:8])

	// Read the rest: headers + payload + 4-byte message CRC.
	remaining := make([]byte, totalLen-12)
	if _, err := io.ReadFull(s.reader, remaining); err != nil {
		return nil, nil, err
	}

	// Parse headers.
	headerBytes := remaining[:headersLen]
	headers = make(map[string]string)
	for len(headerBytes) > 0 {
		if len(headerBytes) < 1 {
			break
		}
		nameLen := int(headerBytes[0])
		headerBytes = headerBytes[1:]
		if len(headerBytes) < nameLen {
			break
		}
		name := string(headerBytes[:nameLen])
		headerBytes = headerBytes[nameLen:]

		if len(headerBytes) < 1 {
			break
		}
		valueType := headerBytes[0]
		headerBytes = headerBytes[1:]

		if valueType == 7 { // string type
			if len(headerBytes) < 2 {
				break
			}
			valueLen := int(binary.BigEndian.Uint16(headerBytes[:2]))
			headerBytes = headerBytes[2:]
			if len(headerBytes) < valueLen {
				break
			}
			headers[name] = string(headerBytes[:valueLen])
			headerBytes = headerBytes[valueLen:]
		}
	}

	// Payload sits between headers and the 4-byte message CRC at the end.
	payloadLen := int(totalLen) - 12 - int(headersLen) - 4
	if payloadLen > 0 {
		payload = remaining[headersLen : headersLen+uint32(payloadLen)]
	}

	return headers, payload, nil
}

// Close closes the underlying response body.
func (s *bedrockStreamReader) Close() error {
	return s.body.Close()
}
