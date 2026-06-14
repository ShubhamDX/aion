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
	return payload
}
