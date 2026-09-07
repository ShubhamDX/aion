package provider

import (
	"encoding/json"

	"github.com/ShubhamDX/aion/internal/types"
)

// bedrockCacheText preserves only the text and explicit cache checkpoint fields.
// Requests without checkpoints retain their existing translation.
func bedrockCacheText(message types.Message) (any, bool) {
	var parts []struct {
		Type         string          `json:"type"`
		Text         string          `json:"text"`
		CacheControl json.RawMessage `json:"cache_control,omitempty"`
	}
	if json.Unmarshal(message.Content, &parts) != nil || len(parts) == 0 {
		return nil, false
	}
	found := false
	for _, part := range parts {
		if part.Type != "text" {
			return nil, false
		}
		if len(part.CacheControl) > 0 && string(part.CacheControl) != "null" {
			found = true
		}
	}
	return parts, found
}

func translateBedrockCacheMessages(source []types.Message, system any) (any, []anthropicMsg) {
	// Translate marked messages independently so preceding merged tool results
	// cannot shift the checkpoint onto a different message.
	var result []anthropicMsg
	var systems []any
	markedSystem := false
	for _, message := range source {
		content, marked := bedrockCacheText(message)
		if message.Role == "system" || message.Role == "developer" {
			if marked {
				var blocks []map[string]any
				encoded, _ := json.Marshal(content)
				_ = json.Unmarshal(encoded, &blocks)
				for _, block := range blocks {
					systems = append(systems, block)
				}
				markedSystem = true
			} else if message.ContentString() != "" {
				systems = append(systems, map[string]string{"type": "text", "text": message.ContentString()})
			}
			continue
		}
		if marked && (message.Role == "user" || (message.Role == "assistant" && len(message.ToolCalls) == 0)) {
			result = append(result, anthropicMsg{Role: message.Role, Content: content})
			continue
		}
		_, translated := translateAnthropicMessages([]types.Message{message})
		for _, item := range translated {
			if blocks, ok := item.Content.([]anthropicContent); ok && allAnthropicToolResults(blocks) {
				for _, block := range blocks {
					appendAnthropicToolResult(&result, block)
				}
			} else {
				result = append(result, item)
			}
		}
	}
	if markedSystem {
		system = systems
	} else if system == "" {
		system = nil
	}
	return system, result
}
