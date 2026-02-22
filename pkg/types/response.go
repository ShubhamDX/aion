package types

// ChatCompletionResponse represents an OpenAI-compatible chat completion response.
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"` // "chat.completion"
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice represents a single completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage reports token consumption for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionChunk represents a single SSE chunk in a streaming response.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"` // "chat.completion.chunk"
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// ChunkChoice represents a single choice within a streaming chunk.
type ChunkChoice struct {
	Index        int        `json:"index"`
	Delta        ChunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"` // nullable
}

// ChunkDelta represents the incremental content in a streaming chunk.
type ChunkDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   *string    `json:"content,omitempty"` // nullable
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ErrorResponse represents an OpenAI-compatible error response.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains the details of an API error.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

// ModelListResponse represents the response from the /v1/models endpoint.
type ModelListResponse struct {
	Object string      `json:"object"` // "list"
	Data   []ModelInfo `json:"data"`
}

// ModelInfo describes a single model in the models list.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "model"
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}
