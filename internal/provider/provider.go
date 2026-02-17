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
