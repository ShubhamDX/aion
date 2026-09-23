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
			// An assistant turn carries no content in the OpenAI wire format
			// when it only issues tool calls, and when it refuses (the text is
			// in refusal instead). Clients spell the empty content either by
			// omitting the field or by sending an explicit null, and both are
			// valid; every other message, including a tool-result reply, must
			// have content.
			if m.Role == "assistant" && (len(m.ToolCalls) > 0 || m.Refusal != nil) {
				continue
			}
			return fmt.Errorf("messages[%d].content is required", i)
		}
		if err := validateContent(fmt.Sprintf("messages[%d].content", i), m.Content); err != nil {
			return err
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
		if err := validateContent(fmt.Sprintf("messages[%d].content", i), m.Content); err != nil {
			return err
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

// validateContent checks that content, already known to be present, is one of
// the two shapes either wire format uses: a JSON string, or a non-empty array
// whose every element is a content block (an object carrying a non-empty
// `type`). A bare number, boolean or object, and an array holding nulls,
// scalars or untyped objects, would otherwise reach the provider unexamined
// and come back as an opaque upstream error. field is the caller's JSON path
// to content, used to build the error message.
func validateContent(field string, content json.RawMessage) error {
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(content, &parts); err != nil || len(parts) == 0 {
		return fmt.Errorf("%s must be a string or an array of content parts, got %s", field, content)
	}
	for j, part := range parts {
		var block struct {
			Type string          `json:"type"`
			Text json.RawMessage `json:"text"`
		}
		if err := json.Unmarshal(part, &block); err != nil || block.Type == "" {
			return fmt.Errorf("%s[%d] must be a content part object with a non-empty type, got %s", field, j, part)
		}
		// Only "text" gets a required-field check here: it is the one block
		// type this proxy reads directly (ContentString, the Anthropic
		// text+tool_use merge). Other known types (tool_use, tool_result,
		// image_url, and any type this proxy does not interpret) are passed
		// through structurally so a valid multimodal, tool, or refusal
		// history is never broken by a field shape this proxy doesn't use.
		if block.Type == "text" {
			var text string
			if len(block.Text) == 0 || json.Unmarshal(block.Text, &text) != nil {
				return fmt.Errorf("%s[%d] is a text block and requires a string-valued text field, got %s", field, j, part)
			}
		}
	}
	return nil
}
