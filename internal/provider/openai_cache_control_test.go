package provider

import (
	"encoding/json"
	"github.com/ShubhamDX/aion/internal/types"
	"strings"
	"testing"
)

func TestOpenAICacheMetadataIsNotSentOrMutated(t *testing.T) {
	raw := json.RawMessage(`[{"type":"text","text":"prefix","cache_control":{"type":"ephemeral"}}]`)
	req := &types.ChatCompletionRequest{Messages: []types.Message{{Role: "system", Content: raw}}}
	for _, stream := range []bool{false, true} {
		payload, _ := openAINativePayload(req, "test", stream)
		if strings.Contains(string(payload.Messages[0].Content), "cache_control") {
			t.Fatal("Anthropic checkpoint sent to native OpenAI")
		}
		if payload.Messages[0].ContentString() != "prefix" {
			t.Fatal("text changed")
		}
		if string(req.Messages[0].Content) != string(raw) {
			t.Fatal("governed request mutated")
		}
	}
}
