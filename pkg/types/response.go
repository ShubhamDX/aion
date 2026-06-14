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

// PromptTokensDetails carries OpenAI-compatible prompt-token subcounts.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// Usage reports token consumption for a request. PromptTokens is total input.
// The input partition is:
//
//	PromptTokens = UncachedInputTokens + CacheReadInputTokens + CacheCreationInputTokens
//
// Providers without cache fields normalize all prompt tokens as uncached.
type Usage struct {
	PromptTokens             int                  `json:"prompt_tokens"`
	CompletionTokens         int                  `json:"completion_tokens"`
	TotalTokens              int                  `json:"total_tokens"`
	PromptTokensDetails      *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	UncachedInputTokens      int                  `json:"uncached_input_tokens,omitempty"`
	CacheReadInputTokens     int                  `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int                  `json:"cache_creation_input_tokens,omitempty"`
	ProviderCacheMode        string               `json:"provider_cache_mode,omitempty"`
}

// NormalizeInputPartition fills missing partition fields without changing the
// customer-facing aggregate token counts.
func (u *Usage) NormalizeInputPartition() {
	if u == nil {
		return
	}
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 && u.CacheReadInputTokens == 0 {
		u.CacheReadInputTokens = u.PromptTokensDetails.CachedTokens
	}
	if u.UncachedInputTokens == 0 {
		if u.CacheReadInputTokens > 0 || u.CacheCreationInputTokens > 0 {
			u.UncachedInputTokens = maxInt(u.PromptTokens-u.CacheReadInputTokens-u.CacheCreationInputTokens, 0)
		} else {
			u.UncachedInputTokens = u.PromptTokens
		}
	}
	input := u.UncachedInputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	if u.PromptTokens == 0 && input > 0 {
		u.PromptTokens = input
	}
	total := u.PromptTokens + u.CompletionTokens
	if total > u.TotalTokens {
		u.TotalTokens = total
	}
}

// InputPartitionValid reports whether the normalized input classes add up to
// the total prompt input and contain no negative values.
func (u Usage) InputPartitionValid() bool {
	u.NormalizeInputPartition()
	if u.PromptTokens < 0 || u.CompletionTokens < 0 || u.TotalTokens < 0 ||
		u.UncachedInputTokens < 0 || u.CacheReadInputTokens < 0 ||
		u.CacheCreationInputTokens < 0 {
		return false
	}
	input := u.UncachedInputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	return u.PromptTokens == input
}

// MergeFrom merges a streaming usage chunk into an accumulated usage value.
func (u *Usage) MergeFrom(chunk Usage) {
	chunk.NormalizeInputPartition()
	if chunk.PromptTokens > 0 {
		u.PromptTokens = chunk.PromptTokens
	}
	if chunk.CompletionTokens > 0 {
		u.CompletionTokens = chunk.CompletionTokens
	}
	if chunk.TotalTokens > 0 {
		u.TotalTokens = chunk.TotalTokens
	}
	if chunk.PromptTokensDetails != nil {
		u.PromptTokensDetails = chunk.PromptTokensDetails
	}
	if chunk.UncachedInputTokens > 0 {
		u.UncachedInputTokens = chunk.UncachedInputTokens
	}
	if chunk.CacheReadInputTokens > 0 {
		u.CacheReadInputTokens = chunk.CacheReadInputTokens
	}
	if chunk.CacheCreationInputTokens > 0 {
		u.CacheCreationInputTokens = chunk.CacheCreationInputTokens
	}
	if chunk.ProviderCacheMode != "" {
		u.ProviderCacheMode = chunk.ProviderCacheMode
	}
	u.NormalizeInputPartition()
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
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
