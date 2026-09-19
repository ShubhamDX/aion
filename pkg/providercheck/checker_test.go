package providercheck

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
)

func TestCheckerClassifiesCloudCredentialFailure(t *testing.T) {
	code, message := classifyError(errors.New("vertex: do request: cloud credential unavailable"))
	if code != "authentication_failed" || message != "The configured cloud identity could not provide a usable token." {
		t.Fatalf("code=%q message=%q", code, message)
	}
}

func TestCheckerDiscardsOutputAndReportsTier(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer provider-secret" {
			t.Fatalf("authorization header was not sent")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "diag", "model": "test-model", "choices": []any{}})
	}))
	defer server.Close()
	cfg := &config.Config{Providers: config.ProvidersConfig{OpenAI: &config.ProviderConfig{
		APIKey: "provider-secret", BaseURL: server.URL,
		Models: []config.ModelConfig{{ID: "test-model", Tier: 2}},
	}}}
	checker, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	result := checker.Test(t.Context(), "openai", "test-model")
	if !result.Connected || result.Tier != 2 || result.Code != "connected" {
		t.Fatalf("result = %#v", result)
	}
}

func TestCheckerSanitizesProviderError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "secret upstream account detail", http.StatusForbidden)
	}))
	defer server.Close()
	cfg := &config.Config{Providers: config.ProvidersConfig{OpenAI: &config.ProviderConfig{
		APIKey: "provider-secret", BaseURL: server.URL,
		Models: []config.ModelConfig{{ID: "test-model", Tier: 1}},
	}}}
	checker, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	result := checker.Test(t.Context(), "openai", "test-model")
	if result.Connected || result.Code != "authentication_failed" {
		t.Fatalf("result = %#v", result)
	}
	if result.Message == "" || result.Message == "secret upstream account detail" {
		t.Fatalf("provider body leaked: %#v", result)
	}
}
