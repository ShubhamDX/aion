package provider

import (
	"context"
	"io"

	"github.com/ShubhamDX/aion/internal/types"
)

// Response wraps the result from a provider.
type Response struct {
	ChatResponse *types.ChatCompletionResponse
	StatusCode   int
	// SchemaNativeEmitted reports whether this provider actually emitted native
	// structured output (OpenAI response_format.json_schema) on the upstream
	// request. It is the wire-truth an embedding product gates its evidence on: it
	// is true ONLY when the schema body was placed on the dispatched payload.
	// Default false (no native output emitted).
	SchemaNativeEmitted bool
}

// StreamReader is an interface for reading streaming chunks.
type StreamReader interface {
	// ReadChunk reads the next chunk from the stream. Returns io.EOF when done.
	ReadChunk() (*types.ChatCompletionChunk, error)
	Close() error
}

// Provider sends requests to an LLM provider.
type Provider interface {
	Name() string
	Send(ctx context.Context, req *types.ChatCompletionRequest, model string) (*Response, error)
	SendStream(ctx context.Context, req *types.ChatCompletionRequest, model string) (StreamReader, error)
}

// ensure StreamReader requires io.EOF sentinel
var _ error = io.EOF

// upstreamPayload builds the request body to send to an OpenAI-compatible
// provider: a copy of req with the resolved model and stream flag set, and ALL
// AION control fields stripped. aion_preferences (incl session_id) is an
// internal routing/session hint, never a provider field; sending it upstream
// would leak the caller's session id and a non-standard field. Translating
// providers (Anthropic/Bedrock/Vertex) build their own payloads and never see
// these fields; the pass-through providers MUST go through this helper so a new
// AION control field can never silently ship upstream.
func upstreamPayload(req *types.ChatCompletionRequest, model string, stream bool) types.ChatCompletionRequest {
	payload := *req
	payload.Model = model
	payload.Stream = stream
	payload.AIONPreferences = nil
	// SchemaSettings is an internal carrier and is `json:"-"`, so it never
	// serializes; nil it here too as defense in depth, mirroring AIONPreferences,
	// so a future struct change can never ship it upstream. The OpenAI-native
	// emission path (openAINativePayload) explicitly re-derives response_format
	// from the carrier BEFORE calling this is irrelevant: it sets ResponseFormat on
	// the returned payload, not the carrier.
	payload.SchemaSettings = nil
	return payload
}

// openAINativePayload builds the OpenAI upstream payload, activating native
// structured output ONLY when the request carries SchemaSettings with
// Mode==provider_native and a schema body. It returns the payload and whether
// native output was emitted (the wire truth reported back via Response).
//
// This is the ONLY place AION sets response_format from a schema policy, and
// only for the OpenAI-compatible provider. Any other mode, a missing body, or a
// caller that already set response_format leaves the payload exactly as
// upstreamPayload produced it (no partial schema, no clobber).
func openAINativePayload(req *types.ChatCompletionRequest, model string, stream bool) (types.ChatCompletionRequest, bool) {
	payload := upstreamPayload(req, model, stream)
	ss := req.SchemaSettings
	if ss == nil || ss.Mode != types.ProviderSchemaModeProviderNative || ss.SchemaBody == nil {
		return payload, false
	}
	// Do not clobber a caller-supplied response_format: if the caller already
	// asked for a specific format, AION does not override it. Native emission only
	// fills an empty slot.
	if payload.ResponseFormat != nil {
		return payload, false
	}
	payload.ResponseFormat = &types.ResponseFormat{
		Type:       "json_schema",
		JSONSchema: ss.SchemaBody,
	}
	return payload, true
}
