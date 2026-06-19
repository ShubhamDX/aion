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

// sendableStub implements provider.Provider with a configurable name and a fixed
// response, so a route override can move the request to a different provider.
type sendableStub struct{ name string }

func (s sendableStub) Name() string { return s.name }
func (s sendableStub) Send(context.Context, *types.ChatCompletionRequest, string) (*provider.Response, error) {
	return &provider.Response{
		StatusCode: http.StatusOK,
		ChatResponse: &types.ChatCompletionResponse{
			Choices: []types.Choice{{Message: types.Message{Role: "assistant", Content: json.RawMessage(`"ok"`)}, FinishReason: "stop"}},
			Usage:   types.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
	}, nil
}
func (s sendableStub) SendStream(context.Context, *types.ChatCompletionRequest, string) (provider.StreamReader, error) {
	return nil, nil
}

// twoProviderHandler registers two providers (openai tier-1, bedrock tier-2) so a
// PreRequest route override can move the final route to a different provider.
func twoProviderHandler(t *testing.T) *Handler {
	t.Helper()
	cfg := &config.Config{}
	cfg.Providers.OpenAI = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "gpt-cheap", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}},
	}
	cfg.Providers.Bedrock = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "claude-strong", Tier: 2, InputPricePer1M: 3, OutputPricePer1M: 6}},
	}
	reg := provider.NewRegistry()
	reg.Register(sendableStub{name: "openai"})
	reg.Register(sendableStub{name: "bedrock"})
	return &Handler{router: router.NewRouter(cfg, nil), registry: reg, pricing: pricing.NewTable(cfg.Providers)}
}

// The post-route seam must receive the FINAL provider/model after a route
// override, not the pre-route selection.
func TestResolveSchemaSettings_SeamSeesFinalRoute(t *testing.T) {
	h := twoProviderHandler(t)
	var sawProvider, sawModel string
	h.SetGatewayHooks(&types.GatewayHooks{
		PreRequest: func(in types.PreRequestInput) types.PreRequestDecision {
			// Override the route to the bedrock model (a different provider).
			return types.PreRequestDecision{Verdict: types.VerdictRoute, RoutedModelOverride: "claude-strong"}
		},
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			sawProvider = in.RoutedProvider
			sawModel = in.RoutedModel
			return &types.SchemaSettings{Mode: "validation_only", SchemaPolicyID: "sp-x"}
		},
	})

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)

	if sawProvider != "bedrock" || sawModel != "claude-strong" {
		t.Fatalf("seam must see the FINAL route, got provider=%q model=%q", sawProvider, sawModel)
	}
	if w.Code != http.StatusOK {
		t.Fatalf("request must still succeed, got %d", w.Code)
	}
}

// The seam must also fire when there is NO PreRequest hook: it is independent of
// PreRequest and only needs the final route.
func TestResolveSchemaSettings_SeamFiresWithoutPreRequest(t *testing.T) {
	h := twoProviderHandler(t)
	called := false
	h.SetGatewayHooks(&types.GatewayHooks{
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			called = true
			if in.RoutedModel != "gpt-cheap" || in.RoutedProvider != "openai" {
				t.Errorf("seam must see the selected route, got %q/%q", in.RoutedProvider, in.RoutedModel)
			}
			return nil
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	if !called {
		t.Fatal("seam must fire without a PreRequest hook")
	}
}
