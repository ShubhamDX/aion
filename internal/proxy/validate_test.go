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
