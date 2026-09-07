package provider

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestBedrockPreservesExplicitCacheCheckpoints(t *testing.T) {
	for _, role := range []string{"system", "developer", "user", "assistant"} {
		t.Run(role, func(t *testing.T) {
			req := &types.ChatCompletionRequest{Messages: []types.Message{
				{Role: role, Content: json.RawMessage(`[{"type":"text","text":"stable prefix","cache_control":{"type":"ephemeral","ttl":"5m"}}]`)},
				{Role: "user", Content: json.RawMessage(`"next"`)},
			}}
			for _, stream := range []bool{false, true} {
				body, err := json.Marshal((&BedrockProvider{}).translateRequest(req, stream))
				if err != nil {
					t.Fatal(err)
				}
				if strings.Count(string(body), `"cache_control"`) != 1 || !strings.Contains(string(body), `"text":"stable prefix"`) {
					t.Fatalf("checkpoint lost: %s", body)
				}
			}
		})
	}
}

func TestBedrockUnmarkedTranslationUnchanged(t *testing.T) {
	messages := []types.Message{
		{Role: "system", Content: json.RawMessage(`"instructions"`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"hello"}]`)},
		{Role: "tool", ToolCallID: "a", Content: json.RawMessage(`"first"`)},
		{Role: "tool", ToolCallID: "b", Content: json.RawMessage(`"second"`)},
	}
	wantSystem, wantMessages := translateAnthropicMessages(messages)
	gotSystem, gotMessages := translateBedrockCacheMessages(messages, wantSystem)
	want, _ := json.Marshal([]any{wantSystem, wantMessages})
	got, _ := json.Marshal([]any{gotSystem, gotMessages})
	if string(got) != string(want) {
		t.Fatalf("changed unmarked request: got %s want %s", got, want)
	}
}

func TestBedrockOmitsEmptySystem(t *testing.T) {
	body, err := json.Marshal((&BedrockProvider{}).translateRequest(&types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}}}, false))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `"system"`) {
		t.Fatalf("empty system was added: %s", body)
	}
}
