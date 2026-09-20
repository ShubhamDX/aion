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
