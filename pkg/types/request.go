package types

import "encoding/json"

// ChatCompletionRequest represents an OpenAI-compatible chat completion request
// with an additional AIONPreferences field for routing hints.
type ChatCompletionRequest struct {
	Model            string           `json:"model"`
	Messages         []Message        `json:"messages"`
	Temperature      *float64         `json:"temperature,omitempty"`
	TopP             *float64         `json:"top_p,omitempty"`
	N                *int             `json:"n,omitempty"`
	Stream           bool             `json:"stream,omitempty"`
	Stop             json.RawMessage  `json:"stop,omitempty"` // string or []string
	MaxTokens        *int             `json:"max_tokens,omitempty"`
	PresencePenalty  *float64         `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64         `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]int   `json:"logit_bias,omitempty"`
	User             string           `json:"user,omitempty"`
	Tools            []Tool           `json:"tools,omitempty"`
	ToolChoice       json.RawMessage  `json:"tool_choice,omitempty"` // string or object
	ResponseFormat   *ResponseFormat  `json:"response_format,omitempty"`
	Seed             *int             `json:"seed,omitempty"`
	AIONPreferences  *AIONPreferences `json:"aion_preferences,omitempty"`
	// SchemaSettings carries the resolved schema-policy provider settings for an
	// embedding product (e.g. AION Enterprise). It is `json:"-"`: it NEVER
	// serializes onto any upstream provider payload. SP2b-2 only carries it; no
	// provider path reads it to mutate a request yet. nil leaves behavior
	// unchanged.
	SchemaSettings *SchemaSettings `json:"-"`
}

// SchemaSettings is the neutral, provider-agnostic carrier for resolved
// schema-policy settings. It holds the resolved provider schema MODE and schema
// IDENTITY metadata only; it contains NO schema body, NO prompt text and NO
// response-format / tool-schema / envelope payload. It is an in-memory carrier
// (request field is `json:"-"`) and must never be serialized to a provider.
//
// SP2b-2 populated and carried it; SP2b-3b is the first slice where a provider
// (OpenAI-compatible only) reads Mode + SchemaBody to emit native structured
// output, gated behind golden tests.
type SchemaSettings struct {
	// Mode is the resolved provider schema mode (validation_only, provider_native,
	// tool_schema, prompt_envelope, none). In observe traffic this is
	// validation_only.
	Mode string
	// SchemaPolicyID / SchemaVersion / SchemaHash are schema IDENTITY only (the
	// same one-way identifiers carried in signed evidence). No raw schema.
	SchemaPolicyID string
	SchemaVersion  string
	SchemaHash     string
	// FailClosed records that a mandatory enforcement could not be honored by the
	// resolved provider path. The gateway acts on it before dispatch for the
	// OpenAI-native scope (SP2b-3b); other scopes still only carry it.
	FailClosed bool
	// Reason is the non-sensitive resolver reason label (for logs / later
	// evidence). No prompt/response/schema content.
	Reason string
	// SchemaBody is the raw JSON Schema the embedding product wants emitted as
	// OpenAI native structured output. It is carried IN MEMORY only (the request
	// field is json:"-"), read solely by the OpenAI-compatible provider path when
	// Mode==provider_native, and MUST NEVER be persisted, logged or signed (AION
	// signs only SchemaHash). nil when no native emission applies.
	SchemaBody *JSONSchemaSpec
	// MustEmitNative marks that native structured output is MANDATORY for this
	// request: the embedding product set a fail-closed policy that the final route
	// can honor. The gateway refuses to serve before dispatch if native emission
	// cannot actually happen (e.g. the caller already set response_format, which
	// the provider will not override), rather than serving an unconstrained
	// response. false leaves native emission best-effort.
	MustEmitNative bool
}

// Message represents a single message in a chat conversation.
// Content is json.RawMessage to support both string and array-of-content-parts
// (multimodal/vision requests use the array form).
type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

// ContentString returns the message content as a plain string.
// If content is a JSON string it is unquoted; if it is an array of
// content parts the text parts are concatenated; otherwise the raw
// bytes are returned as-is.
func (m Message) ContentString() string {
	if len(m.Content) == 0 {
		return ""
	}

	// Try simple string first (most common case).
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}

	// Try array of content parts (vision / multimodal).
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var combined string
		for _, p := range parts {
			if p.Type == "text" {
				combined += p.Text
			}
		}
		return combined
	}

	// Fallback: return raw bytes.
	return string(m.Content)
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	Index    *int         `json:"index,omitempty"`
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

// ResponseFormat specifies the desired response format. Type is the OpenAI
// response_format type ("text", "json_object", "json_schema"). JSONSchema is set
// only for Type=="json_schema" (OpenAI native structured output); it is omitted
// otherwise so an unchanged request marshals exactly as before.
type ResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema *JSONSchemaSpec `json:"json_schema,omitempty"`
}

// JSONSchemaSpec is the OpenAI response_format.json_schema payload: a name, a
// strict flag, and the JSON Schema itself. Schema is the raw schema body; it
// rides only on the in-memory request (never persisted or signed by AION).
type JSONSchemaSpec struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict,omitempty"`
	Schema json.RawMessage `json:"schema"`
}

// ProviderSchemaMode values the provider path compares against. These mirror the
// embedding product's mode strings; only the ones the OSS provider path needs are
// declared here (it activates native output on provider_native only).
const (
	ProviderSchemaModeValidationOnly = "validation_only"
	ProviderSchemaModeProviderNative = "provider_native"
)
