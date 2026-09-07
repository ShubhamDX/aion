package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestBedrockCacheCheckpointAndUsageThroughHTTP(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		if len(body.Messages) != 1 || len(body.Messages[0].Content) != 2 || body.Messages[0].Content[0]["cache_control"] == nil {
			t.Error("checkpoint lost before upstream dispatch")
			w.WriteHeader(400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"test","model":"haiku","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":4,"cache_read_input_tokens":5000}}`))
	}))
	defer upstream.Close()
	p := &BedrockProvider{baseURL: upstream.URL, client: upstream.Client(), bearerToken: "test-only"}
	response, err := p.Send(context.Background(), &types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"prefix","cache_control":{"type":"ephemeral"}},{"type":"text","text":"next"}]`)}}}, "haiku")
	if err != nil {
		t.Fatal(err)
	}
	if response.ChatResponse.Usage.CacheReadInputTokens != 5000 || response.ChatResponse.Usage.UncachedInputTokens != 7 {
		t.Fatalf("cache usage lost: %+v", response.ChatResponse.Usage)
	}
}

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
