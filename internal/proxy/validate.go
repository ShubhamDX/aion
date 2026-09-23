package proxy

import (
	"encoding/json"
	"fmt"

	"github.com/ShubhamDX/aion/internal/types"
)

// ErrNoMessages is returned by validateMessages / validateAnthropicMessages
// when a request has no messages to serve.
var ErrNoMessages = fmt.Errorf("messages is required and must not be empty")

// validateMessages rejects a request before it reaches routing or provider
// dispatch when its message list is structurally unusable (missing, empty,
// or containing a message with no role/content).
//
// A missing or empty `model` field is deliberately NOT an error here: that
// is the documented aion-auto path (see docs/CLIENT_SETUP.md, "Automatic
// routing") and must keep classifying and routing normally.
func validateMessages(messages []types.Message) error {
	if len(messages) == 0 {
		return ErrNoMessages
	}
	for i, m := range messages {
		if m.Role == "" {
			return fmt.Errorf("messages[%d].role is required", i)
		}
		// An assistant turn that only issues tool calls carries no content in
		// the OpenAI wire format; every other message, including a tool-result
		// reply, must have some.
		if m.Role == "assistant" && len(m.ToolCalls) > 0 && len(m.Content) == 0 {
			continue
		}
		if len(m.Content) == 0 {
			return fmt.Errorf("messages[%d].content is required", i)
		}
		if !validContentShape(m.Content) {
			return fmt.Errorf("messages[%d].content must be a string or an array of content parts, got %s", i, m.Content)
		}
	}
	return nil
}

// validateAnthropicMessages is the same structural check for the Anthropic
// ingress message shape, applied before translateAnthropicToOpenAI. Anthropic
// represents tool use as a content block inside the array itself, so unlike
// the OpenAI shape there is no role/tool_calls exception here.
func validateAnthropicMessages(messages []anthropicIngressMsg) error {
	if len(messages) == 0 {
		return ErrNoMessages
	}
	for i, m := range messages {
		if m.Role == "" {
			return fmt.Errorf("messages[%d].role is required", i)
		}
		if len(m.Content) == 0 {
			return fmt.Errorf("messages[%d].content is required", i)
		}
		if !validContentShape(m.Content) {
			return fmt.Errorf("messages[%d].content must be a string or an array of content parts, got %s", i, m.Content)
		}
	}
	return nil
}

// validContentShape reports whether content decodes to a JSON string or a
// non-empty JSON array of content parts, the only two shapes either wire
// format uses. A bare null, number, boolean, or object is not a usable
// message content shape and would otherwise reach the provider unexamined.
func validContentShape(content json.RawMessage) bool {
	// Unmarshaling JSON null into any pointer target succeeds as a no-op in
	// Go, so it must be rejected explicitly before trying the string/array
	// shapes below, or a literal `null` would pass as a valid empty string.
	if string(content) == "null" {
		return false
	}
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return true
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(content, &parts); err == nil {
		return len(parts) > 0
	}
	return false
}
