package proxy

import (
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
		if len(m.Content) == 0 {
			return fmt.Errorf("messages[%d].content is required", i)
		}
	}
	return nil
}

// validateAnthropicMessages is the same structural check for the Anthropic
// ingress message shape, applied before translateAnthropicToOpenAI.
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
	}
	return nil
}
