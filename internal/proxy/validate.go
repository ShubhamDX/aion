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
		if contentAbsent(m.Content) {
			// An assistant turn that only issues tool calls carries no content
			// in the OpenAI wire format. Clients spell that either by omitting
			// the field or by sending an explicit null, and both are valid;
			// every other message, including a tool-result reply, must have
			// content.
			if m.Role == "assistant" && len(m.ToolCalls) > 0 {
				continue
			}
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
		if contentAbsent(m.Content) {
			return fmt.Errorf("messages[%d].content is required", i)
		}
		if !validContentShape(m.Content) {
			return fmt.Errorf("messages[%d].content must be a string or an array of content parts, got %s", i, m.Content)
		}
	}
	return nil
}

// contentAbsent reports whether a message carries no content at all. An
// omitted field and an explicit JSON null mean the same thing on the wire,
// so callers must not treat one as present and the other as missing.
func contentAbsent(content json.RawMessage) bool {
	return len(content) == 0 || string(content) == "null"
}

// validContentShape reports whether content, already known to be present,
// decodes to a JSON string or a non-empty JSON array of content parts, the
// only two shapes either wire format uses. A bare number, boolean, or object
// is not a usable message content shape and would otherwise reach the
// provider unexamined.
func validContentShape(content json.RawMessage) bool {
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
