package types

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// governedMessage is the normalized, governed view of one message. Content is
// the RAW message content bytes (not text-flattened) so multimodal / image /
// file parts change the digest, plus the tool surface (tool calls + the tool a
// message answers).
type governedMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

// governedRequest is the canonical, governed projection of EVERY parsed request
// field that reaches policy, routing, pricing or provider dispatch — not just
// the prompt + tool surface. Two requests that differ in any cost- or
// policy-relevant way (max_tokens, stop, response_format, seed, user,
// aion_preferences, sampling params) MUST produce different digests, so the
// signed evidence binds the full governed input (review finding #1).
type governedRequest struct {
	Model            string            `json:"model"`
	Messages         []governedMessage `json:"messages"`
	Tools            []Tool            `json:"tools,omitempty"`
	ToolChoice       json.RawMessage   `json:"tool_choice,omitempty"`
	MaxTokens        *int              `json:"max_tokens,omitempty"`
	Temperature      *float64          `json:"temperature,omitempty"`
	TopP             *float64          `json:"top_p,omitempty"`
	N                *int              `json:"n,omitempty"`
	Stop             json.RawMessage   `json:"stop,omitempty"`
	PresencePenalty  *float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64          `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]int    `json:"logit_bias,omitempty"`
	User             string            `json:"user,omitempty"`
	ResponseFormat   *ResponseFormat   `json:"response_format,omitempty"`
	Seed             *int              `json:"seed,omitempty"`
	AIONPreferences  *AIONPreferences  `json:"aion_preferences,omitempty"`
	Stream           bool              `json:"stream"`
}

// RequestContentDigest returns a sha256-hex digest over the request's FULL
// governed surface (every parsed field reaching policy/routing/pricing/dispatch:
// model, messages incl raw multimodal content + tool calls, tools, tool_choice,
// max_tokens, stop, response_format, seed, user, sampling params,
// aion_preferences, stream). One-way hash, never the content itself. Go's
// encoding/json emits struct fields in declaration order and map keys sorted, so
// the canonical bytes are stable for equal inputs.
func RequestContentDigest(req *ChatCompletionRequest) string {
	if req == nil {
		return ""
	}
	g := governedRequest{
		Model: req.Model, Tools: req.Tools, ToolChoice: req.ToolChoice,
		MaxTokens: req.MaxTokens, Temperature: req.Temperature, TopP: req.TopP,
		N: req.N, Stop: req.Stop, PresencePenalty: req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty, LogitBias: req.LogitBias,
		User: req.User, ResponseFormat: req.ResponseFormat, Seed: req.Seed,
		AIONPreferences: req.AIONPreferences, Stream: req.Stream,
	}
	for _, m := range req.Messages {
		g.Messages = append(g.Messages, governedMessage{
			Role: m.Role, Content: m.Content, Name: m.Name,
			ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID,
		})
	}
	return digestJSON(g)
}

// governedChoice is the governed projection of one response choice: the RAW
// message content (multimodal-safe) AND any tool calls the model emitted (a
// tool-only response has no text but must still bind its action surface).
type governedChoice struct {
	Content      json.RawMessage `json:"content"`
	ToolCalls    []ToolCall      `json:"tool_calls,omitempty"`
	FinishReason string          `json:"finish_reason"`
}

// ResponseContentDigest returns a sha256-hex digest over the response's governed
// surface: each choice's raw content + tool calls + finish reason. "" when there
// are no choices (e.g. an error or an empty/streamed-unavailable response).
func ResponseContentDigest(resp *ChatCompletionResponse) string {
	if resp == nil || len(resp.Choices) == 0 {
		return ""
	}
	gs := make([]governedChoice, 0, len(resp.Choices))
	for _, c := range resp.Choices {
		gs = append(gs, governedChoice{
			Content: c.Message.Content, ToolCalls: c.Message.ToolCalls,
			FinishReason: c.FinishReason,
		})
	}
	return digestJSON(gs)
}

// ResponseContentStrings returns EVERY choice's assistant message content as a
// slice, in choice order, for in-memory response-shape validation by an
// embedding product. nil when there are no choices. All choices are returned
// (not just the first) so a validator sees every served choice: with n > 1 the
// first choice could pass while a later served one fails. It is the raw content;
// callers must not persist or sign it.
func ResponseContentStrings(resp *ChatCompletionResponse) []string {
	if resp == nil || len(resp.Choices) == 0 {
		return nil
	}
	out := make([]string, 0, len(resp.Choices))
	for _, c := range resp.Choices {
		out = append(out, c.Message.ContentString())
	}
	return out
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
	// ResolveSchemaSettings is an optional post-route seam. The proxy calls it
	// AFTER PreRequest and AFTER any route override is resolved, so the FINAL
	// provider/model is known, but BEFORE provider dispatch. The embedding product
	// resolves the schema-policy provider settings against the final route and
	// returns them; the proxy stamps the result onto req.SchemaSettings (the
	// json:"-" carrier). When this hook is installed, its return value REPLACES
	// any carrier a PreRequest decision set; return nil to clear it. A nil hook
	// (not installed) leaves whatever PreRequest set untouched. No provider reads
	// the carrier yet, so this cannot change a served response or upstream payload.
	ResolveSchemaSettings func(PostRouteInput) *SchemaSettings
	// ApplyContextCompression is an optional post-route, pre-dispatch seam. The
	// proxy calls it AFTER routing, schema resolution and budget, but BEFORE
	// dispatch, on the non-streaming path only. The embedding product may return a
	// replacement message set (a deterministically compressed view of the request
	// it already computed from the ORIGINAL intent); the proxy swaps req.Messages
	// for the returned slice before building the upstream payload. Returning a nil
	// result (or a nil hook) leaves the request unchanged. Routing already happened
	// on the original request, so compression cannot change the routed model. The
	// embedding product owns the no-raw-evidence discipline; the proxy never logs
	// or persists the messages.
	ApplyContextCompression func(PostRouteInput) *ContextCompressionResult
	// ApplyOutputControl is an optional post-context, pre-dispatch seam. The proxy
	// calls it AFTER context compression, schema resolution and budget, but BEFORE
	// dispatch, on the non-streaming path only. The embedding product receives the
	// current in-memory request shape and may return a final message set and/or a
	// final max_tokens cap. Returning nil or a result with no fields set leaves
	// the request unchanged. The hook is for output-token controls such as a fixed
	// style envelope; the proxy never logs or persists the request body.
	ApplyOutputControl func(OutputControlInput) *OutputControlResult
	// ResponseAction is an optional response-boundary seam. When installed, the
	// proxy normalizes every COMPLETE model-proposed tool call in a response
	// (buffering streaming tool-call fragments to completion first) and calls this
	// hook ONCE per response with all proposed calls, BEFORE releasing any tool
	// call to the client. The embedding product returns an allow/block/hold
	// decision per call; the proxy releases the response unchanged only when every
	// call is allowed, and otherwise replaces the tool calls with a protocol-valid
	// completion carrying the action envelope (no executable tool call reaches the
	// client). A nil hook leaves the OSS response bytes unchanged. The hook never
	// receives raw arguments — only a per-call sha256 arguments digest — and no
	// provider credentials. OSS defines the seam and the envelope shape; it does
	// not define policy.
	ResponseAction func(ResponseActionInput) ResponseActionDecision
}

// ContextCompressionResult is the optional replacement the ApplyContextCompression
// seam returns. Messages, when non-nil, replaces req.Messages before dispatch.
// Applied is the embedding product's own record that the swap happened (the proxy
// applies the swap iff Messages is non-nil; Applied is informational for the
// caller's evidence). A nil result or nil Messages means no change.
type ContextCompressionResult struct {
	Messages []Message
	Applied  bool
}

// OutputControlInput is the final-route, post-context view handed to
// ApplyOutputControl. Request is the current parsed request just before provider
// dispatch. It may already include context compression. RequestDigest binds that
// current request shape before output control applies.
type OutputControlInput struct {
	PostRouteInput
	Request *ChatCompletionRequest
}

// OutputControlResult is the optional replacement the ApplyOutputControl seam
// returns. Messages, when non-nil, replaces req.Messages before dispatch.
// MaxTokens, when non-nil, replaces req.MaxTokens before dispatch. Applied is
// informational for the embedding product's evidence; the proxy applies each
// non-nil field regardless of Applied.
type OutputControlResult struct {
	Messages  []Message
	MaxTokens *int
	Applied   bool
}

// MaxBufferedToolCallArgsBytes is the documented ceiling on the buffered
// arguments of a single streaming tool call. A tool call whose accumulated
// arguments exceed it is treated as malformed and fails closed (blocked), so a
// hostile or runaway stream cannot force unbounded buffering when the
// ResponseAction hook is enabled. Non-streaming responses are already bounded by
// the provider read limits.
const MaxBufferedToolCallArgsBytes = 256 * 1024

// ProposedToolCall is one complete model-proposed tool call handed to the
// ResponseAction hook. Args is the raw arguments JSON string ONLY so the
// embedding product can classify (e.g. destructive-action detection); ArgsDigest
// is the sha256-hex the product records as evidence. Index is the tool call's
// position in the response (stable across parallel calls). ID is the provider
// call id.
type ProposedToolCall struct {
	Index      int
	ID         string
	Name       string
	Args       string
	ArgsDigest string
}

// ResponseActionInput is the complete set of proposed tool calls for one
// response, handed to the ResponseAction hook before any is released. It carries
// no provider credentials and no prompt/response text beyond the tool calls the
// client would otherwise execute.
type ResponseActionInput struct {
	RequestID     string
	PrincipalID   string
	RequestDigest string
	Protocol      string // "openai_chat" | "openai_responses" | "anthropic"
	ToolCalls     []ProposedToolCall
}

// ResponseActionVerdict is the per-call decision.
type ResponseActionVerdict int

const (
	// ActionAllow releases the tool call to the client unchanged.
	ActionAllow ResponseActionVerdict = iota
	// ActionBlock refuses the tool call: it never reaches the client.
	ActionBlock
	// ActionHold withholds the tool call pending human approval: it never reaches
	// the client as an executable proposal.
	ActionHold
)

// ResponseActionCallDecision is the decision for one proposed tool call. HoldID
// is a durable identifier the embedding product assigns for a held call so the
// client and operator can correlate the approval; empty for allow/block.
type ResponseActionCallDecision struct {
	Index       int
	Verdict     ResponseActionVerdict
	PolicyClass string
	HoldID      string
}

// ResponseActionDecision is the hook result: one decision per proposed call. The
// proxy releases the response unchanged only when every decision is ActionAllow.
// If any call is blocked or held, NO tool call is released; the proxy emits the
// action envelope instead (all-or-nothing, so a client never receives half an
// executable plan). ReasonCode is surfaced for audit.
type ResponseActionDecision struct {
	Decisions  []ResponseActionCallDecision
	ReasonCode string
}

// AllAllowed reports whether every proposed call was allowed (the response may be
// released unchanged). An empty decision set is treated as all-allowed.
func (d ResponseActionDecision) AllAllowed() bool {
	for _, dec := range d.Decisions {
		if dec.Verdict != ActionAllow {
			return false
		}
	}
	return true
}

// ArgsDigestHex returns the sha256-hex of a tool call's raw arguments string, the
// digest the ResponseAction hook records instead of the raw arguments.
func ArgsDigestHex(args string) string {
	sum := sha256.Sum256([]byte(args))
	return hex.EncodeToString(sum[:])
}

// PostRouteInput is the final-route view handed to ResolveSchemaSettings. The
// provider/model/tier here are the FINAL ones after route override (unlike
// PreRequestInput, which carries the pre-override route). RequestedModel is the
// caller's original model string. It carries no raw bodies; RequestDigest is the
// same one-way digest the other hooks receive.
type PostRouteInput struct {
	RequestID      string
	PrincipalID    string
	RequestedModel string
	// RoutedProvider / RoutedModel / Tier are the FINAL route after any override.
	RoutedProvider string
	RoutedModel    string
	Tier           Tier
	RequestDigest  string
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
	// SessionMaterial carries privacy-preserving session/cache identity as
	// digests only (never the raw session id or prompt text). The embedding
	// product re-keys these into tenant-scoped HMACs. Source is a non-sensitive
	// reason field. S1 populates this; it does not change routing.
	SessionMaterial SessionMaterial
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
// in response headers / logs for auditability. RoutedModelOverride /
// RoutedTierOverride are honored only when Verdict == VerdictRoute.
type PreRequestDecision struct {
	Verdict    PreRequestVerdict
	ReasonCode string
	// RoutedModelOverride pins a specific model ID. When set it wins over
	// RoutedTierOverride. The proxy re-resolves it via the router and fails
	// closed if the model is unknown.
	RoutedModelOverride string
	// RoutedTierOverride routes to a tier rather than a fixed model: the proxy
	// picks the cheapest healthy model in EXACTLY that tier from THIS
	// deployment's catalog (strict, no tier fallback), so each deployment
	// downgrades to its own providers without the policy naming a model ID. 0
	// means unset. Ignored when RoutedModelOverride is set. If the tier has no
	// eligible healthy model the route fails closed (it does NOT widen to
	// another tier).
	RoutedTierOverride int
	// RoutedAllowedProviders, when non-empty, constrains BOTH route forms to
	// models from these providers only. It mirrors the embedding product's
	// provider allowlist: a tier downgrade cannot pick a cheaper model from a
	// disallowed provider, and a pinned RoutedModelOverride is resolved through
	// the allowlist so it cannot land on a disallowed (or ambiguous) provider.
	// Empty means no constraint.
	RoutedAllowedProviders []string
	LedgerRowID            string
	// Message is the client-facing error text on Block / Hold.
	Message string
	// SchemaSettings, when non-nil, is the resolved schema-policy provider
	// settings the embedding product wants carried on the request. The proxy
	// stamps it onto the request's `json:"-"` SchemaSettings field after the hook;
	// SP2b-2 only carries it (no provider activates on it, no payload changes).
	SchemaSettings *SchemaSettings
}

// PostResponseInput is the completed-request view for recording. Tokens + cost
// are final. It carries no raw bodies, but it DOES carry content DIGESTS
// (sha256 hex over the canonical request/response content) so the embedding
// product can compute an evidence anchor that binds the recorded row to the
// actual governed content, not merely to the request id. RequestDigest /
// ResponseDigest are computed by the proxy from the parsed bodies; they are
// one-way hashes, not the content itself.
type PostResponseInput struct {
	RequestID                 string
	PrincipalID               string
	RequestedModel            string
	RoutedModel               string
	RoutedProvider            string
	Tier                      Tier
	InputTokens               int
	OutputTokens              int
	UncachedInputTokens       int
	CacheReadInputTokens      int
	CacheCreationInputTokens  int
	ProviderCacheMode         string
	UncachedInputCostUSD      float64
	CacheReadInputCostUSD     float64
	CacheCreationInputCostUSD float64
	OutputCostUSD             float64
	CostUSD                   float64
	SavingsUSD                float64
	LatencyMS                 int64
	StatusCode                int
	Stream                    bool
	// RequestDigest is sha256-hex over the canonical request content.
	RequestDigest string
	// ResponseDigest is sha256-hex over the canonical response content ("" if
	// the response had no recordable content, e.g. an error).
	ResponseDigest string
	// SessionMaterial carries the same digest-only session/cache identity as the
	// pre-request hook, plus NextCachePrefixMaterialSHA256 (the next turn's prefix
	// digest) when the full response body is available. On streaming the next
	// prefix digest is "" (the stream is not buffered to compute it) and the
	// embedding product safe-degrades. Digests only, never raw material.
	SessionMaterial SessionMaterial
	// ResponseContents is the parsed assistant message content of EVERY response
	// choice (choice order), passed IN MEMORY so an embedding product can validate
	// the response shape (e.g. schema policy) across all served choices, not just
	// the first. It is nil on streaming (not reassembled) and on an empty/error
	// response. The embedding product MUST NOT persist or sign it; it is the live
	// body, the same trust level as PreRequestInput.Request. The proxy never logs
	// or stores it.
	ResponseContents []string
	// SchemaNativeEmitted is the wire truth: true ONLY when the provider actually
	// emitted native structured output (OpenAI response_format.json_schema) on the
	// dispatched request. An embedding product gates its evidence on this so it
	// cannot claim provider_native when the resolver selected it but the wire path
	// did not emit it. false on streaming and on every non-OpenAI path.
	SchemaNativeEmitted bool
}

// PreRequestInput.RequestDigest mirror: the pre-request hook also needs the
// request content digest so a blocked/routed decision can be anchored to the
// governed prompt. It is added to PreRequestInput below via the RequestDigest
// field.
