package proxy

import (
	"context"
	"encoding/json"
	"fmt"
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

func TestValidateMessages(t *testing.T) {
	refusalText := "I can't help with that."
	cases := []struct {
		name    string
		msgs    []types.Message
		wantErr bool
	}{
		{"nil messages", nil, true},
		{"empty messages", []types.Message{}, true},
		{"missing role", []types.Message{{Content: json.RawMessage(`"hi"`)}}, true},
		{"missing content", []types.Message{{Role: "user"}}, true},
		{"valid", []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}}, false},
		{
			"assistant tool-call-only message has no content",
			[]types.Message{
				{Role: "user", Content: json.RawMessage(`"what's the weather?"`)},
				{Role: "assistant", ToolCalls: []types.ToolCall{{ID: "call_1", Type: "function", Function: types.FunctionCall{Name: "get_weather", Arguments: `{"city":"nyc"}`}}}},
				{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage(`"72F and sunny"`)},
			},
			false,
		},
		{
			"assistant message with neither content nor tool_calls is still invalid",
			[]types.Message{{Role: "assistant"}},
			true,
		},
		{"content is bare null", []types.Message{{Role: "user", Content: json.RawMessage(`null`)}}, true},
		{"content is a bare number", []types.Message{{Role: "user", Content: json.RawMessage(`123`)}}, true},
		{"content is a bare object", []types.Message{{Role: "user", Content: json.RawMessage(`{}`)}}, true},
		{"content is an empty array", []types.Message{{Role: "user", Content: json.RawMessage(`[]`)}}, true},
		{"content is a non-empty content-part array", []types.Message{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"hi"}]`)}}, false},
		{
			"assistant tool-call-only message with explicit null content",
			[]types.Message{
				{Role: "user", Content: json.RawMessage(`"what's the weather?"`)},
				{Role: "assistant", Content: json.RawMessage(`null`), ToolCalls: []types.ToolCall{{ID: "call_1", Type: "function", Function: types.FunctionCall{Name: "get_weather", Arguments: `{"city":"nyc"}`}}}},
				{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage(`"72F and sunny"`)},
			},
			false,
		},
		{
			"assistant message with explicit null content and no tool_calls is still invalid",
			[]types.Message{{Role: "assistant", Content: json.RawMessage(`null`)}},
			true,
		},
		{
			"assistant refusal turn with null content",
			[]types.Message{
				{Role: "user", Content: json.RawMessage(`"do something disallowed"`)},
				{Role: "assistant", Content: json.RawMessage(`null`), Refusal: &refusalText},
			},
			false,
		},
		{
			"assistant refusal turn with omitted content",
			[]types.Message{
				{Role: "user", Content: json.RawMessage(`"do something disallowed"`)},
				{Role: "assistant", Refusal: &refusalText},
			},
			false,
		},
		{"content array holds a null element", []types.Message{{Role: "user", Content: json.RawMessage(`[null]`)}}, true},
		{"content array holds a bare number", []types.Message{{Role: "user", Content: json.RawMessage(`[123]`)}}, true},
		{"content array holds an object with no type", []types.Message{{Role: "user", Content: json.RawMessage(`[{}]`)}}, true},
		{"content array holds a valid part followed by a malformed one", []types.Message{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"hi"},123]`)}}, true},
		{"text block missing the text field", []types.Message{{Role: "user", Content: json.RawMessage(`[{"type":"text"}]`)}}, true},
		{"text block with a non-string text field", []types.Message{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":123}]`)}}, true},
		{"text block with a null text field", []types.Message{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":null}]`)}}, true},
		{"non-text block with no text field is untouched", []types.Message{{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"t1","name":"f","input":{}}]`)}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMessages(tc.msgs)
			if (err != nil) != tc.wantErr {
				t.Fatalf("validateMessages(%+v) error=%v, wantErr=%v", tc.msgs, err, tc.wantErr)
			}
		})
	}
}

func TestValidateAnthropicMessages(t *testing.T) {
	cases := []struct {
		name    string
		msgs    []anthropicIngressMsg
		wantErr bool
	}{
		{"nil messages", nil, true},
		{"empty messages", []anthropicIngressMsg{}, true},
		{"missing role", []anthropicIngressMsg{{Content: json.RawMessage(`"hi"`)}}, true},
		{"missing content", []anthropicIngressMsg{{Role: "user"}}, true},
		{"valid", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`"hi"`)}}, false},
		{"content is bare null", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`null`)}}, true},
		{"content is a bare number", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`123`)}}, true},
		{"content is a bare object", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`{}`)}}, true},
		{"content is an empty array", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`[]`)}}, true},
		{"content is a non-empty content-block array", []anthropicIngressMsg{{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"t1","name":"get_weather","input":{}}]`)}}, false},
		{"content array holds a null block", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`[null]`)}}, true},
		{"content array holds a bare number", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`[123]`)}}, true},
		{"content array holds a block with no type", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`[{}]`)}}, true},
		{"text block missing the text field", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`[{"type":"text"}]`)}}, true},
		{"text block with a non-string text field", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":123}]`)}}, true},
		{"text block with a null text field", []anthropicIngressMsg{{Role: "user", Content: json.RawMessage(`[{"type":"text","text":null}]`)}}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAnthropicMessages(tc.msgs)
			if (err != nil) != tc.wantErr {
				t.Fatalf("validateAnthropicMessages(%+v) error=%v, wantErr=%v", tc.msgs, err, tc.wantErr)
			}
		})
	}
}

// TestChatCompletionRejectsInvalidInputBeforeDispatch proves invalid-input
// requests never reach routing/provider dispatch: the Handler here has a
// nil router, classifier and registry, so touching any of them would panic.
// A clean 400 response means validation short-circuited first — zero
// upstream calls were possible.
func TestChatCompletionRejectsInvalidInputBeforeDispatch(t *testing.T) {
	h := &Handler{}

	cases := []struct {
		name string
		body string
	}{
		{"missing messages field", `{"model":"some-model"}`},
		{"empty messages array", `{"model":"some-model","messages":[]}`},
		{"message with no role", `{"model":"some-model","messages":[{"content":"hi"}]}`},
		{"message with no content", `{"model":"some-model","messages":[{"role":"user"}]}`},
		{"content is bare null", `{"model":"some-model","messages":[{"role":"user","content":null}]}`},
		{"content is a bare number", `{"model":"some-model","messages":[{"role":"user","content":123}]}`},
		{"content is a bare object", `{"model":"some-model","messages":[{"role":"user","content":{}}]}`},
		{"content is an empty array", `{"model":"some-model","messages":[{"role":"user","content":[]}]}`},
		{"content array holds a null element", `{"model":"some-model","messages":[{"role":"user","content":[null]}]}`},
		{"content array holds a bare number", `{"model":"some-model","messages":[{"role":"user","content":[123]}]}`},
		{"content array holds an object with no type", `{"model":"some-model","messages":[{"role":"user","content":[{}]}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()

			h.ChatCompletion(rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusBadRequest, rec.Body.String())
			}
			var resp struct {
				Error struct {
					Type string `json:"type"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
				t.Fatalf("decode response: %v; body=%s", err, rec.Body.String())
			}
			if resp.Error.Type != "invalid_request" {
				t.Fatalf("error.type = %q, want invalid_request", resp.Error.Type)
			}
		})
	}
}

// TestChatCompletionPreservesAutoRoutingOnMissingModel proves a missing
// `model` field is NOT rejected by input validation — it must still reach
// routing (the documented aion-auto path) and dispatch to a provider exactly
// once, using a functioning fake classifier and provider rather than
// inferring success from a panic against nil dependencies.
func TestChatCompletionPreservesAutoRoutingOnMissingModel(t *testing.T) {
	calls := 0
	h := handlerWithClassifierAndCallCounter(&calls)
	body := `{"messages":[{"role":"user","content":"hi"}]}` // no "model" field
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.ChatCompletion(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if calls != 1 {
		t.Fatalf("provider was called %d times, want 1 (missing `model` must still dispatch via aion-auto)", calls)
	}
}

// TestChatCompletionAcceptsToolCallContinuation proves a valid multi-turn
// tool-calling conversation (assistant message with tool_calls, followed by
// a tool-result reply) is NOT rejected by validation and dispatches exactly
// once, whether the client omits the assistant message's content or sends it
// as an explicit null.
func TestChatCompletionAcceptsToolCallContinuation(t *testing.T) {
	cases := []struct {
		name         string
		assistantMsg string
	}{
		{"content omitted", `{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"nyc\"}"}}]}`},
		{"content is explicit null", `{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"nyc\"}"}}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			h := handlerWithCallCounter(&calls)
			body := `{"model":"haiku","messages":[
				{"role":"user","content":"what's the weather in nyc?"},
				` + tc.assistantMsg + `,
				{"role":"tool","tool_call_id":"call_1","content":"72F and sunny"}
			]}`
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			rec := httptest.NewRecorder()

			h.ChatCompletion(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
			}
			if calls != 1 {
				t.Fatalf("provider was called %d times, want 1", calls)
			}
		})
	}
}

// countingProvider is a functioning fake provider (unlike the nil-dependency
// Handler used elsewhere in this file) that records how many times it was
// actually called, so a test can assert zero dispatch rather than infer it
// from a panic.
type countingProvider struct{ calls *int }

func (countingProvider) Name() string { return "bedrock" }
func (p countingProvider) Send(context.Context, *types.ChatCompletionRequest, string) (*provider.Response, error) {
	*p.calls++
	return &provider.Response{
		StatusCode: http.StatusOK,
		ChatResponse: &types.ChatCompletionResponse{
			Choices: []types.Choice{{Message: types.Message{Role: "assistant", Content: json.RawMessage(`"ok"`)}, FinishReason: "stop"}},
			Usage:   types.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		},
	}, nil
}
func (p countingProvider) SendStream(context.Context, *types.ChatCompletionRequest, string) (provider.StreamReader, error) {
	*p.calls++
	return nil, fmt.Errorf("countingProvider: streaming not used by these tests")
}

// handlerWithCallCounter builds a Handler with a real router and a real
// (counting) provider, model "haiku" routable end to end, so a test can prove
// a request either never reached the provider or reached it exactly once.
func handlerWithCallCounter(calls *int) *Handler {
	cfg := &config.Config{}
	cfg.Providers.Bedrock = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "haiku", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}},
	}
	reg := provider.NewRegistry()
	reg.Register(countingProvider{calls: calls})
	return &Handler{router: router.NewRouter(cfg, nil), registry: reg, pricing: pricing.NewTable(cfg.Providers)}
}

// handlerWithClassifierAndCallCounter is handlerWithCallCounter plus a
// working classifier (fixedTierClassifier, defined in
// post_route_seam_test.go), routable end to end through the aion-auto path.
func handlerWithClassifierAndCallCounter(calls *int) *Handler {
	h := handlerWithCallCounter(calls)
	h.classifier = fixedTierClassifier{tier: types.Tier1}
	return h
}

// TestChatCompletionRejectsMalformedTypedContentBlocks proves the two
// required-field gaps found in review (a "text" block missing its text
// field, and one whose text field is not a string) are rejected with 400 and
// never reach the provider, using a functioning fake provider rather than a
// nil-dependency panic.
func TestChatCompletionRejectsMalformedTypedContentBlocks(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"text block missing the text field", `{"model":"haiku","messages":[{"role":"user","content":[{"type":"text"}]}]}`},
		{"text block with a non-string text field", `{"model":"haiku","messages":[{"role":"user","content":[{"type":"text","text":123}]}]}`},
		{"text block with a null text field", `{"model":"haiku","messages":[{"role":"user","content":[{"type":"text","text":null}]}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			h := handlerWithCallCounter(&calls)
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()

			h.ChatCompletion(rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusBadRequest, rec.Body.String())
			}
			if calls != 0 {
				t.Fatalf("provider was called %d times, want 0", calls)
			}
		})
	}
}

// TestChatCompletionAcceptsValidTypedContentBlocks proves a well-formed text
// content block reaches the provider exactly once, so the stricter check
// added for malformed blocks does not also reject the valid shape.
func TestChatCompletionAcceptsValidTypedContentBlocks(t *testing.T) {
	calls := 0
	h := handlerWithCallCounter(&calls)
	body := `{"model":"haiku","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.ChatCompletion(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if calls != 1 {
		t.Fatalf("provider was called %d times, want 1", calls)
	}
}

// TestAnthropicMessagesRejectsInvalidInputBeforeDispatch mirrors the
// ChatCompletion case for the /v1/messages ingress.
func TestAnthropicMessagesRejectsInvalidInputBeforeDispatch(t *testing.T) {
	h := &Handler{}

	cases := []struct {
		name string
		body string
	}{
		{"missing messages field", `{"model":"some-model","max_tokens":100}`},
		{"empty messages array", `{"model":"some-model","max_tokens":100,"messages":[]}`},
		{"content is bare null", `{"model":"some-model","max_tokens":100,"messages":[{"role":"user","content":null}]}`},
		{"content is a bare number", `{"model":"some-model","max_tokens":100,"messages":[{"role":"user","content":123}]}`},
		{"content array holds a null block", `{"model":"some-model","max_tokens":100,"messages":[{"role":"user","content":[null]}]}`},
		{"content array holds an object with no type", `{"model":"some-model","max_tokens":100,"messages":[{"role":"user","content":[{}]}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()

			h.AnthropicMessages(rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusBadRequest, rec.Body.String())
			}
			var resp struct {
				Type  string `json:"type"`
				Error struct {
					Type string `json:"type"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
				t.Fatalf("decode response: %v; body=%s", err, rec.Body.String())
			}
			if resp.Error.Type != "invalid_request_error" {
				t.Fatalf("error.type = %q, want invalid_request_error", resp.Error.Type)
			}
		})
	}
}
