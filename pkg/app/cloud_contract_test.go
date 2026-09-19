package app

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
)

type offlineCloudRoundTripper struct {
	base http.RoundTripper
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
					io.WriteString(w, `{"id":"blocked","content":[],"stop_reason":"refusal","usage":{"input_tokens":12,"output_tokens":0}}`)
				} else {
					if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
						t.Error("wrong compatible endpoint")
					}
					io.WriteString(w, `{"id":"blocked","model":"cloud-fixture","choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"Fixture refusal."},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":0,"total_tokens":12}}`)
				}
			}))
			defer upstream.Close()
			cfg := &config.Config{}
			pc := &config.ProviderConfig{APIKey: "local-test-only", BaseURL: upstream.URL,
				ProjectID: "test-project", Region: "us-central1",
				Models: []config.ModelConfig{{ID: "cloud-fixture", Tier: 3}}}
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
			a, err := Build(Options{Config: cfg})
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
			if response.Code != 200 || calls != 1 {
				t.Fatalf("gateway dispatch: status=%d calls=%d body=%s", response.Code, calls, response.Body.String())
			}
			body := response.Body.String()
			if (providerName == "vertex" && !strings.Contains(body, `"finish_reason":"refusal"`)) ||
				(providerName != "vertex" && !strings.Contains(body, `"refusal":"Fixture refusal."`)) {
				t.Fatal("gateway hid provider refusal")
			}
		})
	}
}
