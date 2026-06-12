package types

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// governedMessage is the normalized, governed view of one message: role, text
// content, and the tool surface (tool calls the model requested + the tool the
// message answers). Tool calls + tool_call_id are part of the governed action
// surface for agent traffic, so they are bound into the digest.
type governedMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// governedRequest is the canonical, governed projection of a request that the
// digest covers: model, every message (incl tool calls), the available tools,
// and the tool_choice. This binds the ACTION surface, not just the prompt text,
// so a signed row proves what tools/actions were in scope (review finding #3).
type governedRequest struct {
	Model      string            `json:"model"`
	Messages   []governedMessage `json:"messages"`
	Tools      []Tool            `json:"tools,omitempty"`
	ToolChoice json.RawMessage   `json:"tool_choice,omitempty"`
}

// RequestContentDigest returns a sha256-hex digest over the request's full
// governed surface (model, messages, tool calls, tools, tool_choice). It is a
// one-way hash, never the content itself. Determinism: Go's encoding/json emits
// struct fields in declaration order and map keys sorted, so the canonical bytes
// are stable for equal inputs.
func RequestContentDigest(req *ChatCompletionRequest) string {
	if req == nil {
		return ""
	}
	g := governedRequest{Model: req.Model, Tools: req.Tools, ToolChoice: req.ToolChoice}
	for _, m := range req.Messages {
		g.Messages = append(g.Messages, governedMessage{
			Role: m.Role, Content: m.ContentString(),
			ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID,
		})
	}
	return digestJSON(g)
}

// governedChoice is the governed projection of one response choice: the message
// text AND any tool calls the model emitted (a tool-only response has no text
// but must still bind its action surface).
type governedChoice struct {
	Content      string     `json:"content"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
	FinishReason string     `json:"finish_reason"`
}

// ResponseContentDigest returns a sha256-hex digest over the response's governed
// surface: each choice's text + tool calls + finish reason. "" when there are no
// choices (e.g. an error or an empty/streamed-unavailable response).
func ResponseContentDigest(resp *ChatCompletionResponse) string {
	if resp == nil || len(resp.Choices) == 0 {
		return ""
	}
	gs := make([]governedChoice, 0, len(resp.Choices))
	for _, c := range resp.Choices {
		gs = append(gs, governedChoice{
			Content: c.Message.ContentString(), ToolCalls: c.Message.ToolCalls,
			FinishReason: c.FinishReason,
		})
	}
	return digestJSON(gs)
}

// digestJSON marshals v to canonical JSON and returns its sha256 hex.
func digestJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// GatewayHooks is the optional, generic extension surface for wrapping the proxy
// request lifecycle. It is defined in OSS-stable types only (no enterprise
// dependency); an embedding product (e.g. AION Enterprise) populates it to run a
// pre-request decision and a post-response record. Both fields are optional: a
// nil GatewayHooks, or nil func fields, leave the OSS proxy behavior unchanged.
//
// The proxy calls PreRequest AFTER it has parsed + classified the request and
// selected a model, but BEFORE dispatching upstream. It calls PostResponse AFTER
// the upstream response and cost computation, on both the streaming and
// non-streaming paths. Neither hook ever receives raw provider credentials; the
// embedding product computes its own evidence anchors from the data here.
type GatewayHooks struct {
	PreRequest   func(PreRequestInput) PreRequestDecision
	PostResponse func(PostResponseInput)
}

// PreRequestInput is the classified, routed view of a request before dispatch.
// RequestedModel is what the caller asked for ("" / "aion-auto" means
// classify); RoutedModel/RoutedProvider/Tier are what the router chose. Request
// is the parsed body (no provider credentials). PrincipalID identifies the
// caller (the validated API key id, or "" when auth is disabled).
type PreRequestInput struct {
	RequestID      string
	PrincipalID    string
	Request        *ChatCompletionRequest
	RequestedModel string
	RoutedModel    string
	RoutedProvider string
	Tier           Tier
	EstimatedCost  float64
	// RequestDigest is sha256-hex over the canonical request content, so a
	// decision is anchored to the governed prompt (not just the request id).
	RequestDigest string
}

// PreRequestVerdict is the decision the embedding product returns.
type PreRequestVerdict int

const (
	// VerdictAllow lets the request proceed to the routed model unchanged.
	VerdictAllow PreRequestVerdict = iota
	// VerdictRoute proceeds but overrides the model (cheaper-safe routing). The
	// proxy re-resolves RoutedModelOverride before dispatch.
	VerdictRoute
	// VerdictBlock refuses the request (e.g. budget exceeded). The proxy returns
	// an error to the caller and does not dispatch.
	VerdictBlock
	// VerdictHold queues the request for human approval and does not dispatch.
	VerdictHold
)

// PreRequestDecision is the hook result. ReasonCode + LedgerRowID are surfaced
// in response headers / logs for auditability. RoutedModelOverride is honored
// only when Verdict == VerdictRoute.
type PreRequestDecision struct {
	Verdict             PreRequestVerdict
	ReasonCode          string
	RoutedModelOverride string
	LedgerRowID         string
	// Message is the client-facing error text on Block / Hold.
	Message string
}

// PostResponseInput is the completed-request view for recording. Tokens + cost
// are final. It carries no raw bodies, but it DOES carry content DIGESTS
// (sha256 hex over the canonical request/response content) so the embedding
// product can compute an evidence anchor that binds the recorded row to the
// actual governed content, not merely to the request id. RequestDigest /
// ResponseDigest are computed by the proxy from the parsed bodies; they are
// one-way hashes, not the content itself.
type PostResponseInput struct {
	RequestID      string
	PrincipalID    string
	RequestedModel string
	RoutedModel    string
	RoutedProvider string
	Tier           Tier
	InputTokens    int
	OutputTokens   int
	CostUSD        float64
	SavingsUSD     float64
	LatencyMS      int64
	StatusCode     int
	Stream         bool
	// RequestDigest is sha256-hex over the canonical request content.
	RequestDigest string
	// ResponseDigest is sha256-hex over the canonical response content ("" if
	// the response had no recordable content, e.g. an error).
	ResponseDigest string
}

// PreRequestInput.RequestDigest mirror: the pre-request hook also needs the
// request content digest so a blocked/routed decision can be anchored to the
// governed prompt. It is added to PreRequestInput below via the RequestDigest
// field.
