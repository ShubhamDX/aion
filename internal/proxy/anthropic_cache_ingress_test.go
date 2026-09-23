package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/router"
)

func TestAnthropicIngressPreservesTextCacheCheckpoints(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, role := range []string{"system", "user", "assistant"} {
			t.Run(role+map[bool]string{false: "/nonstream", true: "/stream"}[stream], func(t *testing.T) {
				marked := json.RawMessage(`[{"type":"text","text":"prefix","cache_control":{"type":"ephemeral"}},{"type":"text","text":"suffix"}]`)
				raw := map[string]any{"model": "test", "max_tokens": 8, "stream": stream, "messages": []map[string]any{{"role": "user", "content": "next"}}}
				if role == "system" {
					raw["system"] = marked
				} else {
					raw["messages"] = []map[string]any{{"role": role, "content": marked}, {"role": "user", "content": "next"}}
				}
				encoded, _ := json.Marshal(raw)
				var in anthropicIngressRequest
				if err := json.Unmarshal(encoded, &in); err != nil {
					t.Fatal(err)
				}
				out := translateAnthropicToOpenAI(&in)
				if !strings.Contains(string(out.Messages[0].Content), `"cache_control"`) {
					t.Fatalf("checkpoint lost at ingress: %s", out.Messages[0].Content)
				}
				if out.Messages[0].ContentString() != "prefixsuffix" {
					t.Fatalf("text changed: %q", out.Messages[0].ContentString())
				}
			})
		}
	}
}

// TestAnthropicIngressPreservesTrailingTextAfterToolResult proves a
// tool_result turn's trailing text block reaches the translated request as
// its own message, not just the token estimate: the translator used to
// recognize the first block as tool_result and then only ever emit
// tool_result blocks, silently dropping any other block type in that same
// message from the request actually sent to the provider.
func TestAnthropicIngressPreservesTrailingTextAfterToolResult(t *testing.T) {
	raw := map[string]any{"model": "test", "max_tokens": 8, "messages": []map[string]any{
		{"role": "user", "content": []map[string]any{
			{"type": "tool_result", "tool_use_id": "t1", "content": "72F and sunny"},
			{"type": "text", "text": "given that, what should I wear?"},
		}},
	}}
	encoded, _ := json.Marshal(raw)
	var in anthropicIngressRequest
	if err := json.Unmarshal(encoded, &in); err != nil {
		t.Fatal(err)
	}
	out := translateAnthropicToOpenAI(&in)

	if len(out.Messages) != 2 {
		t.Fatalf("got %d messages, want 2 (tool result + trailing text): %+v", len(out.Messages), out.Messages)
	}
	if out.Messages[0].Role != "tool" || out.Messages[0].ToolCallID != "t1" {
		t.Fatalf("messages[0] = %+v, want the tool_result message", out.Messages[0])
	}
	if out.Messages[1].Role != "user" || out.Messages[1].ContentString() != "given that, what should I wear?" {
		t.Fatalf("messages[1] = %+v, want the trailing text preserved as its own user message", out.Messages[1])
	}
}

func TestAnthropicIngressCacheCheckpointReachesBedrockHTTP(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Count(string(body), `"cache_control"`) != 2 {
			t.Error("system or message checkpoint lost before Bedrock")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"test","model":"test","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":4,"cache_read_input_tokens":5000}}`)
	}))
	defer upstream.Close()
	pc := &config.ProviderConfig{APIKey: "test-only", BaseURL: upstream.URL, Models: []config.ModelConfig{{ID: "test", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 5}}}
	cfg := &config.Config{Providers: config.ProvidersConfig{Bedrock: pc}}
	reg := provider.NewRegistry()
	bedrock, err := provider.NewBedrock(pc)
	if err != nil {
		t.Fatal(err)
	}
	reg.Register(bedrock)
	h := NewHandler(nil, router.NewRouter(cfg, nil), reg, nil, pricing.NewTable(cfg.Providers), nil)
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"test","max_tokens":8,"system":[{"type":"text","text":"system prefix","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"user prefix","cache_control":{"type":"ephemeral"}}]}]}`))
	w := httptest.NewRecorder()
	h.AnthropicMessages(w, r)
	if w.Code != 200 {
		t.Fatalf("HTTP %d: %s", w.Code, w.Body.String())
	}
	var response anthropicIngressResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Usage.CacheReadInputTokens != 5000 || response.Usage.InputTokens != 7 {
		t.Fatalf("cache usage lost at client boundary: %+v", response.Usage)
	}
}
