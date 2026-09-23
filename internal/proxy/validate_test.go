package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestValidateMessages(t *testing.T) {
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
// routing (the documented aion-auto path). A nil classifier/router here
// means it panics if reached with no messages error first; we only assert
// it does NOT return the validation-layer 400, confirming the two codepaths
// are distinct.
func TestChatCompletionPreservesAutoRoutingOnMissingModel(t *testing.T) {
	defer func() {
		// A panic here (nil classifier/router) is expected once messages
		// validation passes and routing is attempted — that proves this
		// request was NOT rejected by input validation, which is the point
		// of this test. Swallow it so the test doesn't fail on the panic.
		recover()
	}()
	h := &Handler{}
	body := `{"messages":[{"role":"user","content":"hi"}]}` // no "model" field
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.ChatCompletion(rec, req)

	if rec.Code == http.StatusBadRequest {
		var resp struct {
			Error struct {
				Type string `json:"type"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err == nil && resp.Error.Type == "invalid_request" {
			t.Fatalf("missing `model` was rejected by input validation; it must fall through to aion-auto routing instead")
		}
	}
}

// TestChatCompletionAcceptsToolCallContinuation proves a valid multi-turn
// tool-calling conversation (assistant message with tool_calls and no
// content, followed by a tool-result reply) is NOT rejected by validation.
// A nil classifier/router means reaching dispatch panics; recovering from
// that panic is how this test confirms validation let the request through.
func TestChatCompletionAcceptsToolCallContinuation(t *testing.T) {
	defer func() { recover() }()
	h := &Handler{}
	body := `{"model":"some-model","messages":[
		{"role":"user","content":"what's the weather in nyc?"},
		{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"nyc\"}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"72F and sunny"}
	]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.ChatCompletion(rec, req)

	if rec.Code == http.StatusBadRequest {
		var resp struct {
			Error struct {
				Type string `json:"type"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err == nil && resp.Error.Type == "invalid_request" {
			t.Fatalf("valid tool-call continuation was rejected by input validation: %s", rec.Body.String())
		}
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
