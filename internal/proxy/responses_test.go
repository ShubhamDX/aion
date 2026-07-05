package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestResponses_UsesChatLifecycle(t *testing.T) {
	h := twoProviderHandler(t)
	called := false
	h.SetGatewayHooks(&types.GatewayHooks{
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			called = true
			if in.RoutedModel != "gpt-cheap" || in.RoutedProvider != "openai" {
				t.Errorf("route = %s/%s, want openai/gpt-cheap", in.RoutedProvider, in.RoutedModel)
			}
			return nil
		},
	})

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"gpt-cheap","input":"summarize this","max_output_tokens":64}`))
	h.Responses(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", w.Code, w.Body.String())
	}
	if !called {
		t.Fatal("responses endpoint must use the chat lifecycle hooks")
	}
	var resp struct {
		Object string `json:"object"`
		Output []struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v\n%s", err, w.Body.String())
	}
	if resp.Object != "response" {
		t.Fatalf("object = %q", resp.Object)
	}
	if got := resp.Output[0].Content[0].Text; got != "ok" {
		t.Fatalf("output text = %q", got)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestResponses_RejectsStreaming(t *testing.T) {
	h := twoProviderHandler(t)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"gpt-cheap","input":"hi","stream":true}`))
	h.Responses(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "streaming /v1/responses is not supported") {
		t.Fatalf("missing streaming error: %s", w.Body.String())
	}
}
