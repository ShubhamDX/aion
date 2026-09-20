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

type fixedTierClassifier struct{ tier types.Tier }

func (c fixedTierClassifier) Classify(*types.ChatCompletionRequest) (types.Tier, float64, map[string]float64) {
	return c.tier, 1, nil
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

func fallbackTierHandler(t *testing.T) *Handler {
	t.Helper()
	cfg := &config.Config{}
	cfg.Providers.OpenAI = &config.ProviderConfig{
		Models: []config.ModelConfig{
			{ID: "gpt-cheap", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2},
			{ID: "gpt-premium", Tier: 3, InputPricePer1M: 10, OutputPricePer1M: 30},
		},
	}
	reg := provider.NewRegistry()
	reg.Register(sendableStub{name: "openai"})
	return NewHandler(
		fixedTierClassifier{tier: types.Tier2},
		router.NewRouter(cfg, nil),
		reg,
		nil,
		pricing.NewTable(cfg.Providers),
		nil,
	)
}

func TestTierFallbackReportsSelectedTierOnBothIngresses(t *testing.T) {
	tests := []struct {
		name string
		path string
		body string
		run  func(*Handler, http.ResponseWriter, *http.Request)
	}{
		{
			name: "openai",
			path: "/v1/chat/completions",
			body: `{"model":"aion-auto","messages":[{"role":"user","content":"summarize this"}]}`,
			run:  (*Handler).ChatCompletion,
		},
		{
			name: "anthropic",
			path: "/v1/messages",
			body: `{"model":"aion-auto","max_tokens":16,"messages":[{"role":"user","content":"summarize this"}]}`,
			run:  (*Handler).AnthropicMessages,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := fallbackTierHandler(t)
			var preRequestTier, postRouteTier types.Tier
			h.SetGatewayHooks(&types.GatewayHooks{
				PreRequest: func(in types.PreRequestInput) types.PreRequestDecision {
					preRequestTier = in.Tier
					if in.RoutedModel != "gpt-premium" {
						t.Errorf("pre-request model = %q, want gpt-premium", in.RoutedModel)
					}
					return types.PreRequestDecision{Verdict: types.VerdictAllow}
				},
				ResolveSchemaSettings: func(in types.PostRouteInput) *types.SchemaSettings {
					postRouteTier = in.Tier
					return nil
				},
			})

			w := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			tt.run(h, w, r)

			if w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
			}
			if got := w.Header().Get("X-AION-Model"); got != "gpt-premium" {
				t.Fatalf("X-AION-Model = %q, want gpt-premium", got)
			}
			if got := w.Header().Get("X-AION-Tier"); got != "3" {
				t.Fatalf("X-AION-Tier = %q, want 3", got)
			}
			if preRequestTier != types.Tier3 {
				t.Fatalf("pre-request tier = %d, want 3", preRequestTier)
			}
			if postRouteTier != types.Tier3 {
				t.Fatalf("post-route tier = %d, want 3", postRouteTier)
			}
		})
	}
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
	name   string
	got    *[]types.Message
	gotReq *types.ChatCompletionRequest
}

func (p capturingProvider) Name() string { return p.name }
func (p capturingProvider) Send(_ context.Context, req *types.ChatCompletionRequest, _ string) (*provider.Response, error) {
	if p.got != nil {
		*p.got = req.Messages
	}
	if p.gotReq != nil {
		copied := *req
		copied.Messages = append([]types.Message(nil), req.Messages...)
		*p.gotReq = copied
	}
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
	var req types.ChatCompletionRequest
	return captureRequestHandler(t, got, &req)
}

func captureRequestHandler(t *testing.T, got *[]types.Message, gotReq *types.ChatCompletionRequest) *Handler {
	t.Helper()
	cfg := &config.Config{}
	cfg.Providers.OpenAI = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "gpt-cheap", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}},
	}
	reg := provider.NewRegistry()
	reg.Register(capturingProvider{name: "openai", got: got, gotReq: gotReq})
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

func TestApplyOutputControl_NilHookLeavesRequestUnchanged(t *testing.T) {
	var dispatched []types.Message
	var dispatchedReq types.ChatCompletionRequest
	h := captureRequestHandler(t, &dispatched, &dispatchedReq)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","max_tokens":128,"messages":[{"role":"user","content":"original body"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
	if len(dispatched) != 1 || dispatched[0].ContentString() != "original body" {
		t.Fatalf("nil hook must leave messages unchanged, got %+v", dispatched)
	}
	if dispatchedReq.MaxTokens == nil || *dispatchedReq.MaxTokens != 128 {
		t.Fatalf("nil hook must leave max_tokens unchanged, got %+v", dispatchedReq.MaxTokens)
	}
}

func TestApplyOutputControl_NilResultLeavesRequestUnchanged(t *testing.T) {
	var dispatched []types.Message
	var dispatchedReq types.ChatCompletionRequest
	h := captureRequestHandler(t, &dispatched, &dispatchedReq)
	h.SetGatewayHooks(&types.GatewayHooks{
		ApplyOutputControl: func(in types.OutputControlInput) *types.OutputControlResult {
			if in.Request == nil || len(in.Request.Messages) != 1 || in.Request.Messages[0].ContentString() != "original body" {
				t.Fatalf("output seam saw wrong request: %+v", in.Request)
			}
			return nil
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","max_tokens":128,"messages":[{"role":"user","content":"original body"}]}`))
	h.ChatCompletion(w, r)
	if len(dispatched) != 1 || dispatched[0].ContentString() != "original body" {
		t.Fatalf("nil result must leave messages unchanged, got %+v", dispatched)
	}
	if dispatchedReq.MaxTokens == nil || *dispatchedReq.MaxTokens != 128 {
		t.Fatalf("nil result must leave max_tokens unchanged, got %+v", dispatchedReq.MaxTokens)
	}
}

func TestApplyOutputControl_SwapsMessagesAndMaxTokens(t *testing.T) {
	var dispatched []types.Message
	var dispatchedReq types.ChatCompletionRequest
	h := captureRequestHandler(t, &dispatched, &dispatchedReq)
	finalCap := 64
	h.SetGatewayHooks(&types.GatewayHooks{
		ApplyOutputControl: func(in types.OutputControlInput) *types.OutputControlResult {
			if in.RoutedProvider != "openai" || in.RoutedModel != "gpt-cheap" {
				t.Fatalf("output seam must see final route, got %q/%q", in.RoutedProvider, in.RoutedModel)
			}
			return &types.OutputControlResult{
				Messages:  []types.Message{{Role: "system", Content: json.RawMessage(`"style envelope"`)}, {Role: "user", Content: json.RawMessage(`"final body"`)}},
				MaxTokens: &finalCap,
				Applied:   true,
			}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","max_tokens":128,"messages":[{"role":"user","content":"original body"}]}`))
	h.ChatCompletion(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", w.Code)
	}
	if len(dispatched) != 2 || dispatched[0].ContentString() != "style envelope" || dispatched[1].ContentString() != "final body" {
		t.Fatalf("output seam must replace dispatched messages, got %+v", dispatched)
	}
	if dispatchedReq.MaxTokens == nil || *dispatchedReq.MaxTokens != 64 {
		t.Fatalf("output seam must replace max_tokens, got %+v", dispatchedReq.MaxTokens)
	}
}

func TestApplyOutputControl_RunsAfterContextCompression(t *testing.T) {
	var dispatched []types.Message
	h := captureHandler(t, &dispatched)
	h.SetGatewayHooks(&types.GatewayHooks{
		ApplyContextCompression: func(in types.PostRouteInput) *types.ContextCompressionResult {
			return &types.ContextCompressionResult{
				Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"compressed"`)}},
				Applied:  true,
			}
		},
		ApplyOutputControl: func(in types.OutputControlInput) *types.OutputControlResult {
			if in.Request == nil || len(in.Request.Messages) != 1 || in.Request.Messages[0].ContentString() != "compressed" {
				t.Fatalf("output seam must see the post-compression request, got %+v", in.Request)
			}
			return &types.OutputControlResult{
				Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"output controlled"`)}},
				Applied:  true,
			}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"original body"}]}`))
	h.ChatCompletion(w, r)
	if len(dispatched) != 1 || dispatched[0].ContentString() != "output controlled" {
		t.Fatalf("output seam must run after compression and before dispatch, got %+v", dispatched)
	}
}

func TestApplyOutputControl_AnthropicIngress(t *testing.T) {
	var dispatched []types.Message
	var dispatchedReq types.ChatCompletionRequest
	h := captureRequestHandler(t, &dispatched, &dispatchedReq)
	finalCap := 32
	h.SetGatewayHooks(&types.GatewayHooks{
		ApplyOutputControl: func(in types.OutputControlInput) *types.OutputControlResult {
			if in.RequestedModel != "gpt-cheap" || in.RoutedProvider != "openai" || in.RoutedModel != "gpt-cheap" {
				t.Fatalf("output seam saw wrong route: %+v", in.PostRouteInput)
			}
			return &types.OutputControlResult{
				Messages:  []types.Message{{Role: "user", Content: json.RawMessage(`"anthropic output controlled"`)}},
				MaxTokens: &finalCap,
				Applied:   true,
			}
		},
	})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"gpt-cheap","max_tokens":128,"messages":[{"role":"user","content":"original anthropic body"}]}`))
	h.AnthropicMessages(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("want 200, got %d (%s)", w.Code, w.Body.String())
	}
	if len(dispatched) != 1 || dispatched[0].ContentString() != "anthropic output controlled" {
		t.Fatalf("anthropic ingress must apply output control, got %+v", dispatched)
	}
	if dispatchedReq.MaxTokens == nil || *dispatchedReq.MaxTokens != 32 {
		t.Fatalf("anthropic ingress must apply output max_tokens, got %+v", dispatchedReq.MaxTokens)
	}
}
