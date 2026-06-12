package types

import (
	"crypto/sha256"
	"encoding/hex"
)

// RequestContentDigest returns a sha256-hex digest over the request's governed
// content: the model + each message role and text content, length-framed so two
// different message sets cannot collide. It is a one-way hash of the prompt, not
// the prompt itself; the embedding product anchors evidence to it without ever
// storing raw content.
func RequestContentDigest(req *ChatCompletionRequest) string {
	if req == nil {
		return ""
	}
	h := sha256.New()
	writeField(h, req.Model)
	for _, m := range req.Messages {
		writeField(h, m.Role)
		writeField(h, m.ContentString())
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ResponseContentDigest returns a sha256-hex digest over the response's governed
// content: each choice's message text. "" when there is no content.
func ResponseContentDigest(resp *ChatCompletionResponse) string {
	if resp == nil || len(resp.Choices) == 0 {
		return ""
	}
	h := sha256.New()
	for _, c := range resp.Choices {
		writeField(h, c.Message.ContentString())
	}
	return hex.EncodeToString(h.Sum(nil))
}

// writeField length-frames a string into the hash so concatenation is
// unambiguous (len-prefix prevents "ab"+"c" colliding with "a"+"bc").
func writeField(h interface{ Write([]byte) (int, error) }, s string) {
	var lenBuf [8]byte
	n := uint64(len(s))
	for i := 0; i < 8; i++ {
		lenBuf[i] = byte(n >> (8 * (7 - i)))
	}
	_, _ = h.Write(lenBuf[:])
	_, _ = h.Write([]byte(s))
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
