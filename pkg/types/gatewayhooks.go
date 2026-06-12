package types

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
// are final. It carries no raw bodies; the embedding product computes evidence
// anchors from RequestID + the request/response it already holds.
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
}
