package types

import "encoding/json"

// ChatCompletionRequest represents an OpenAI-compatible chat completion request
// with an additional AIONPreferences field for routing hints.
type ChatCompletionRequest struct {
	Model            string            `json:"model"`
	Messages         []Message         `json:"messages"`
	Temperature      *float64          `json:"temperature,omitempty"`
	TopP             *float64          `json:"top_p,omitempty"`
	N                *int              `json:"n,omitempty"`
	Stream           bool              `json:"stream,omitempty"`
	Stop             json.RawMessage   `json:"stop,omitempty"` // string or []string
	MaxTokens        *int              `json:"max_tokens,omitempty"`
	PresencePenalty  *float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64          `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]int    `json:"logit_bias,omitempty"`
	User             string            `json:"user,omitempty"`
	Tools            []Tool            `json:"tools,omitempty"`
	ToolChoice       json.RawMessage   `json:"tool_choice,omitempty"` // string or object
	ResponseFormat   *ResponseFormat   `json:"response_format,omitempty"`
	Seed             *int              `json:"seed,omitempty"`
	AIONPreferences  *AIONPreferences  `json:"aion_preferences,omitempty"`
}

// Message represents a single message in a chat conversation.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents the function name and arguments in a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool describes a tool available to the model.
type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef describes a function that can be called by the model.
type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ResponseFormat specifies the desired response format.
type ResponseFormat struct {
	Type string `json:"type"`
}
