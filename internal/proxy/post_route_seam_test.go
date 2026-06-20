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
func (s sendableStub) Send(_ context.Context, req *types.ChatCompletionRequest, _ string) (*provider.Response, error) {
	// Mimic the real OpenAI provider: it reports SchemaNativeEmitted=true when the
	// carrier asked for provider_native with a body. A non-openai stub never does
	// (other providers ignore the carrier).
	emitted := s.name == "openai" && req.SchemaSettings != nil &&
		req.SchemaSettings.Mode == types.ProviderSchemaModeProviderNative && req.SchemaSettings.SchemaBody != nil
	return &provider.Response{
		StatusCode: http.StatusOK,
		ChatResponse: &types.ChatCompletionResponse{
			Choices: []types.Choice{{Message: types.Message{Role: "assistant", Content: json.RawMessage(`"ok"`)}, FinishReason: "stop"}},
			Usage:   types.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
		SchemaNativeEmitted: emitted,
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

// SP2b-3b: a mandatory fail-closed schema resolution returns a schema error
// BEFORE dispatch (the request never reaches the provider).
func TestSchemaFailClosed_BlocksBeforeDispatch(t *testing.T) {
	h := twoProviderHandler(t)
	h.SetGatewayHooks(&types.GatewayHooks{
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			return &types.SchemaSettings{Mode: "validation_only", FailClosed: true, SchemaPolicyID: "sp-x"}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusUnprocessableEntity {
		t.Fatalf("mandatory fail-closed must return 422 before dispatch, got %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "schema_policy_unsatisfiable") {
		t.Fatalf("must return the schema error, got %s", w.Body.String())
	}
}

// SP2b-3b: a mandatory native policy (MustEmitNative) whose emission is blocked
// by a caller-supplied response_format must fail closed before dispatch, rather
// than silently serving an unconstrained response.
func TestSchemaFailClosed_MustEmitNativeBlockedByCallerFormat(t *testing.T) {
	h := twoProviderHandler(t)
	h.SetGatewayHooks(&types.GatewayHooks{
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			return &types.SchemaSettings{
				Mode: "provider_native", SchemaPolicyID: "sp-x", MustEmitNative: true,
				SchemaBody: &types.JSONSchemaSpec{Name: "orders", Schema: json.RawMessage(`{"type":"object"}`)},
			}
		},
	})
	w := httptest.NewRecorder()
	// Caller sets response_format -> provider would skip native emission.
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","response_format":{"type":"json_object"},"messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusUnprocessableEntity {
		t.Fatalf("mandatory native blocked by caller format must fail closed, got %d", w.Code)
	}
}

// SP2b-3b: mandatory native on a STREAMING request must fail closed before
// dispatch. A stream could emit native output, but AION cannot reassemble it to
// validate, so there would be native emission with no signed evidence.
func TestSchemaFailClosed_MustEmitNativeBlocksStreaming(t *testing.T) {
	h := twoProviderHandler(t)
	h.SetGatewayHooks(&types.GatewayHooks{
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			return &types.SchemaSettings{
				Mode: "provider_native", SchemaPolicyID: "sp-x", MustEmitNative: true,
				SchemaBody: &types.JSONSchemaSpec{Name: "orders", Schema: json.RawMessage(`{"type":"object"}`)},
			}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusUnprocessableEntity {
		t.Fatalf("mandatory native on a stream must fail closed, got %d", w.Code)
	}
}

// An OPTIONAL native stream (not mandatory) is NOT blocked by the fail-closed
// gate: schemaFailClosed returns false, so the request proceeds to dispatch
// (native is best-effort; no evidence gap to enforce). Asserted at the gate
// rather than end-to-end (the streaming dispatch needs provider stream plumbing).
func TestSchemaFailClosed_OptionalNativeStreamNotGated(t *testing.T) {
	req := &types.ChatCompletionRequest{
		Model:  "gpt-cheap",
		Stream: true,
		SchemaSettings: &types.SchemaSettings{
			Mode: "provider_native", SchemaPolicyID: "sp-x", MustEmitNative: false,
			SchemaBody: &types.JSONSchemaSpec{Name: "orders", Schema: json.RawMessage(`{"type":"object"}`)},
		},
	}
	if schemaFailClosed(req) {
		t.Fatal("optional native stream must NOT be gated")
	}
}

// Mandatory native with NO caller response_format dispatches (emission happens).
func TestSchemaFailClosed_MustEmitNativeDispatchesWhenEmittable(t *testing.T) {
	h := twoProviderHandler(t)
	h.SetGatewayHooks(&types.GatewayHooks{
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			return &types.SchemaSettings{
				Mode: "provider_native", SchemaPolicyID: "sp-x", MustEmitNative: true,
				SchemaBody: &types.JSONSchemaSpec{Name: "orders", Schema: json.RawMessage(`{"type":"object"}`)},
			}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("mandatory native with emittable route must dispatch, got %d", w.Code)
	}
}

// An optional (non-fail-closed) resolution dispatches normally.
func TestSchemaFailClosed_OptionalDispatches(t *testing.T) {
	h := twoProviderHandler(t)
	h.SetGatewayHooks(&types.GatewayHooks{
		ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
			return &types.SchemaSettings{Mode: "validation_only", FailClosed: false, SchemaPolicyID: "sp-x"}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("optional resolution must dispatch, got %d", w.Code)
	}
}

// SP2b-3b confirmation: when the seam carries provider_native + schema body and
// the FINAL route is the OpenAI-compatible provider, the post-response hook sees
// SchemaNativeEmitted=true (the wire truth). A route AWAY from openai must NOT
// confirm native.
func TestSchemaNativeEmitted_ConfirmationFlows(t *testing.T) {
	body := &types.JSONSchemaSpec{Name: "orders", Schema: json.RawMessage(`{"type":"object"}`)}

	// Final route = openai (the stub Name "openai"): native emitted, confirmed.
	t.Run("openai final route confirms", func(t *testing.T) {
		h := twoProviderHandler(t)
		var emitted bool
		var ran bool
		h.SetGatewayHooks(&types.GatewayHooks{
			ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
				return &types.SchemaSettings{Mode: "provider_native", SchemaPolicyID: "sp-x", SchemaBody: body}
			},
			PostResponse: func(in types.PostResponseInput) { ran = true; emitted = in.SchemaNativeEmitted },
		})
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
		h.ChatCompletion(w, r)
		h.DrainPostResponse()
		if !ran || !emitted {
			t.Fatalf("openai native must confirm emission, ran=%v emitted=%v", ran, emitted)
		}
	})

	// Route to bedrock: a non-OpenAI provider never emits native, so no confirm.
	t.Run("non-openai final route does not confirm", func(t *testing.T) {
		h := twoProviderHandler(t)
		var emitted, ran bool
		h.SetGatewayHooks(&types.GatewayHooks{
			PreRequest: func(in types.PreRequestInput) types.PreRequestDecision {
				return types.PreRequestDecision{Verdict: types.VerdictRoute, RoutedModelOverride: "claude-strong"}
			},
			ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
				return &types.SchemaSettings{Mode: "provider_native", SchemaPolicyID: "sp-x", SchemaBody: body}
			},
			PostResponse: func(in types.PostResponseInput) { ran = true; emitted = in.SchemaNativeEmitted },
		})
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
		h.ChatCompletion(w, r)
		h.DrainPostResponse()
		if !ran {
			t.Fatal("post-response must run")
		}
		if emitted {
			t.Fatal("non-openai route must NOT confirm native emission")
		}
	})
}

// capturingProvider records the messages it was asked to dispatch, so a test can
// assert whether the context-compression seam swapped them.
type capturingProvider struct {
	name string
	got  *[]types.Message
}

func (p capturingProvider) Name() string { return p.name }
func (p capturingProvider) Send(_ context.Context, req *types.ChatCompletionRequest, _ string) (*provider.Response, error) {
	*p.got = req.Messages
	return &provider.Response{
		StatusCode: http.StatusOK,
		ChatResponse: &types.ChatCompletionResponse{
			Choices: []types.Choice{{Message: types.Message{Role: "assistant", Content: json.RawMessage(`"ok"`)}, FinishReason: "stop"}},
			Usage:   types.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
	}, nil
}
func (p capturingProvider) SendStream(context.Context, *types.ChatCompletionRequest, string) (provider.StreamReader, error) {
	return nil, nil
}

func captureHandler(t *testing.T, got *[]types.Message) *Handler {
	t.Helper()
	cfg := &config.Config{}
	cfg.Providers.OpenAI = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "gpt-cheap", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}},
	}
	reg := provider.NewRegistry()
	reg.Register(capturingProvider{name: "openai", got: got})
	return &Handler{router: router.NewRouter(cfg, nil), registry: reg, pricing: pricing.NewTable(cfg.Providers)}
}

// CP4 seam: a returned message set replaces req.Messages before dispatch.
func TestApplyContextCompression_SwapsDispatchedMessages(t *testing.T) {
	var dispatched []types.Message
	h := captureHandler(t, &dispatched)
	compressed := []types.Message{{Role: "user", Content: json.RawMessage(`"compressed"`)}}
	h.SetGatewayHooks(&types.GatewayHooks{
		ApplyContextCompression: func(in types.PostRouteInput) *types.ContextCompressionResult {
			return &types.ContextCompressionResult{Messages: compressed, Applied: true}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"original long body"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
	if len(dispatched) != 1 || dispatched[0].ContentString() != "compressed" {
		t.Fatalf("dispatched messages must be the compressed set, got %+v", dispatched)
	}
}

// A nil result leaves the request unchanged.
func TestApplyContextCompression_NilResultLeavesOriginal(t *testing.T) {
	var dispatched []types.Message
	h := captureHandler(t, &dispatched)
	h.SetGatewayHooks(&types.GatewayHooks{
		ApplyContextCompression: func(in types.PostRouteInput) *types.ContextCompressionResult {
			return nil
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"original body"}]}`))
	h.ChatCompletion(w, r)
	if len(dispatched) != 1 || dispatched[0].ContentString() != "original body" {
		t.Fatalf("nil result must leave the original messages, got %+v", dispatched)
	}
}

// CP4 parity: the Anthropic non-stream ingress also applies the context
// compression seam (swaps dispatched messages).
func TestApplyContextCompression_AnthropicIngress(t *testing.T) {
	var dispatched []types.Message
	h := captureHandler(t, &dispatched)
	compressed := []types.Message{{Role: "user", Content: json.RawMessage(`"compressed"`)}}
	h.SetGatewayHooks(&types.GatewayHooks{
		ApplyContextCompression: func(in types.PostRouteInput) *types.ContextCompressionResult {
			return &types.ContextCompressionResult{Messages: compressed, Applied: true}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"gpt-cheap","max_tokens":16,"messages":[{"role":"user","content":"original anthropic body"}]}`))
	h.AnthropicMessages(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d (%s)", w.Code, w.Body.String())
	}
	if len(dispatched) != 1 || dispatched[0].ContentString() != "compressed" {
		t.Fatalf("anthropic ingress must apply the compression seam, got %+v", dispatched)
	}
}
