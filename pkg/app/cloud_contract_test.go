package app

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	pkgtypes "github.com/ShubhamDX/aion/pkg/types"
)

type offlineCloudRoundTripper struct {
	base http.RoundTripper
}

func TestCloudOfflineGatewayInterruptedStreamInvalidatesWarmth(t *testing.T) {
	original := http.DefaultTransport
	http.DefaultTransport = offlineCloudRoundTripper{base: original}
	defer func() { http.DefaultTransport = original }()
	for _, name := range []string{"openai", "gemini", "vertex"} {
		t.Run(name, func(t *testing.T) {
			calls := 0
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls++
				w.Header().Set("Content-Type", "text/event-stream")
				if name == "vertex" {
					io.WriteString(w, "event: message_start\ndata: {\"message\":{\"id\":\"fixture\",\"usage\":{\"input_tokens\":10}}}\n\n")
					io.WriteString(w, "event: content_block_delta\ndata: {\"delta\":{\"type\":\"text_delta\",\"text\":\"12\"}}\n\n")
					io.WriteString(w, "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n")
				} else {
					io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"12\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n")
				}
				// Deliberately close before the protocol's terminal marker.
			}))
			defer upstream.Close()
			cfg := &config.Config{}
			pc := &config.ProviderConfig{APIKey: "synthetic-only", BaseURL: upstream.URL,
				ProjectID: "fixture", Region: "us-central1",
				Models: []config.ModelConfig{{ID: "fixture", Tier: 1}}}
			switch name {
			case "openai":
				cfg.Providers.OpenAI = pc
			case "gemini":
				cfg.Providers.Gemini = pc
			case "vertex":
				cfg.Providers.Vertex = pc
			}
			cfg.Telemetry.DBPath = filepath.Join(t.TempDir(), "telemetry.db")
			var result pkgtypes.PostResponseInput
			a, err := Build(Options{Config: cfg, GatewayHooks: &pkgtypes.GatewayHooks{
				PostResponse: func(in pkgtypes.PostResponseInput) { result = in },
			}})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { a.recorder.Stop(); a.cancel(); a.store.Close() }()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
				strings.NewReader(`{"model":"fixture","stream":true,"messages":[{"role":"user","content":"Return 12."}]}`))
			recorder := httptest.NewRecorder()
			a.Handler().ServeHTTP(recorder, req)
			a.DrainPostResponse()
			if recorder.Code != 200 || calls != 1 || !strings.Contains(recorder.Body.String(), `"code":"upstream_stream_error"`) {
				t.Fatalf("stream failure not surfaced: %d %d %s", recorder.Code, calls, recorder.Body)
			}
			if result.SessionMaterial.NextCachePrefixMaterialSHA256 != "" {
				t.Fatal("interrupted stream contributed cache warmth")
			}
		})
	}
}

func (t offlineCloudRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	ip := net.ParseIP(r.URL.Hostname())
	if r.URL.Scheme != "http" || ip == nil || !ip.IsLoopback() {
		return nil, fmt.Errorf("offline gateway test blocked destination %s", r.URL.Host)
	}
	return t.base.RoundTrip(r)
}

func TestCloudOfflineGatewayConfigurationAndDispatch(t *testing.T) {
	original := http.DefaultTransport
	http.DefaultTransport = offlineCloudRoundTripper{base: original}
	defer func() { http.DefaultTransport = original }()
	for _, providerName := range []string{"openai", "gemini", "vertex"} {
		t.Run(providerName, func(t *testing.T) {
			calls := 0
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body map[string]json.RawMessage
				json.NewDecoder(r.Body).Decode(&body)
				if r.Header.Get("Authorization") != "Bearer local-test-only" || body["tool_choice"] == nil {
					t.Error("gateway did not retain credentials or named tool")
				}
				if providerName == "vertex" {
					if !strings.HasSuffix(r.URL.Path, "/publishers/anthropic/models/cloud-fixture:rawPredict") {
						t.Error("wrong Vertex path")
					}
					io.WriteString(w, `{"id":"blocked","content":[],"stop_reason":"refusal","usage":{"input_tokens":30,"output_tokens":2,"cache_read_input_tokens":50,"cache_creation_input_tokens":20}}`)
				} else {
					if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
						t.Error("wrong compatible endpoint")
					}
					io.WriteString(w, `{"id":"blocked","model":"cloud-fixture","choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"Fixture refusal."},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102,"prompt_tokens_details":{"cached_tokens":50,"cache_write_tokens":20}}}`)
				}
			}))
			defer upstream.Close()
			cfg := &config.Config{}
			pc := &config.ProviderConfig{APIKey: "local-test-only", BaseURL: upstream.URL,
				ProjectID: "test-project", Region: "us-central1",
				Models: []config.ModelConfig{{ID: "cloud-fixture", Tier: 3, InputPricePer1M: 2,
					OutputPricePer1M: 4, CachedInputPricePer1M: .2, CacheWritePricePer1M: 2.5}}}
			switch providerName {
			case "openai":
				pc.BaseURL += "/openai/v1"
				cfg.Providers.OpenAI = pc
			case "gemini":
				pc.BaseURL += "/v1/projects/test-project/locations/us-central1/endpoints/openapi"
				cfg.Providers.Gemini = pc
			case "vertex":
				cfg.Providers.Vertex = pc
			}
			cfg.Telemetry.DBPath = filepath.Join(t.TempDir(), "telemetry.db")
			var observed pkgtypes.PostResponseInput
			a, err := Build(Options{Config: cfg, GatewayHooks: &pkgtypes.GatewayHooks{
				PostResponse: func(in pkgtypes.PostResponseInput) { observed = in },
			}})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				a.DrainPostResponse()
				a.recorder.Stop()
				a.cancel()
				a.store.Close()
			}()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
				`{"model":"cloud-fixture","messages":[{"role":"user","content":"Synthetic contract check."}],"max_tokens":10,"tools":[{"type":"function","function":{"name":"submit_decision","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"submit_decision"}}}`))
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			a.Handler().ServeHTTP(response, request)
			a.DrainPostResponse()
			if response.Code != 200 || calls != 1 {
				t.Fatalf("gateway dispatch: status=%d calls=%d body=%s", response.Code, calls, response.Body.String())
			}
			body := response.Body.String()
			if (providerName == "vertex" && !strings.Contains(body, `"finish_reason":"refusal"`)) ||
				(providerName != "vertex" && !strings.Contains(body, `"refusal":"Fixture refusal."`)) {
				t.Fatal("gateway hid provider refusal")
			}
			if observed.InputTokens != 100 || observed.CacheReadInputTokens != 50 ||
				observed.CacheCreationInputTokens != 20 || math.Abs(observed.CostUSD-.000128) > 1e-12 {
				t.Fatalf("refusal usage or cache cost changed: %+v", observed)
			}
		})
	}
}
