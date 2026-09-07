package provider

import (
	"encoding/json"
	"github.com/ShubhamDX/aion/internal/types"
)

// Anthropic cache checkpoints have no meaning on the native OpenAI endpoint.
// Remove only that field on a private copy, preserving the governed request.
func withoutTextCacheControls(messages []types.Message) []types.Message {
	result := append([]types.Message(nil), messages...)
	for i, m := range messages {
		var blocks []map[string]json.RawMessage
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		changed := false
		for _, b := range blocks {
			if string(b["type"]) == `"text"` && b["cache_control"] != nil {
				delete(b, "cache_control")
				changed = true
			}
		}
		if changed {
			result[i].Content, _ = json.Marshal(blocks)
		}
	}
	return result
}
