package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/types"
)

// stubProvider returns a fixed non-stream response.
type stubProvider struct{ content string }

func (stubProvider) Name() string { return "bedrock" }
func (p stubProvider) Send(context.Context, *types.ChatCompletionRequest, string) (*provider.Response, error) {
	return &provider.Response{
		StatusCode: http.StatusOK,
		ChatResponse: &types.ChatCompletionResponse{
			Choices: []types.Choice{{
				Message:      types.Message{Role: "assistant", Content: json.RawMessage(`"` + p.content + `"`)},
				FinishReason: "stop",
			}},
			Usage: types.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
	}, nil
}
func (stubProvider) SendStream(context.Context, *types.ChatCompletionRequest, string) (provider.StreamReader, error) {
	return nil, nil
}

// orderWriter records, the first time the hook observes it, whether the response
// body had already been written.
type orderWriter struct {
	*httptest.ResponseRecorder
	written bool
}

func (w *orderWriter) Write(b []byte) (int, error) {
	w.written = true
	return w.ResponseRecorder.Write(b)
}

func stubHandler(content string) *Handler {
	cfg := &config.Config{}
	cfg.Providers.Bedrock = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "haiku", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}},
	}
	reg := provider.NewRegistry()
	reg.Register(stubProvider{content: content})
	return &Handler{router: router.NewRouter(cfg, nil), registry: reg, pricing: pricing.NewTable(cfg.Providers)}
}

// The non-stream PostResponse hook must fire AFTER the response body is written,
// so an embedding product's post-response work (evidence, schema validation)
// cannot delay or block the served response.
func TestChatCompletion_PostResponseRunsAfterWrite(t *testing.T) {
	h := stubHandler("hello")
	w := &orderWriter{ResponseRecorder: httptest.NewRecorder()}
	hookRan := false
	bodyWrittenBeforeHook := false
	h.SetGatewayHooks(&types.GatewayHooks{
		PostResponse: func(in types.PostResponseInput) {
			hookRan = true
			bodyWrittenBeforeHook = w.written // must already be true
			if len(in.ResponseContents) != 1 || in.ResponseContents[0] != "hello" {
				t.Errorf("hook must see all parsed choice contents, got %v", in.ResponseContents)
			}
		},
	})

	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"haiku","messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)

	if !hookRan {
		t.Fatal("PostResponse hook did not run")
	}
	if !bodyWrittenBeforeHook {
		t.Fatal("PostResponse must run AFTER the response body is written")
	}
	if w.Body.Len() == 0 {
		t.Fatal("response body must be served")
	}
}
