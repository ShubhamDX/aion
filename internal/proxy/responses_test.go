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

func TestResponses_StreamingEmitsCodexEvents(t *testing.T) {
	h := stubHandler("hello from aion")
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"haiku","input":"hi","stream":true}`))
	h.Responses(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", w.Code, w.Body.String())
	}
	for _, want := range []string{"response.created", "response.output_item.done", "hello from aion", "response.completed"} {
		if !strings.Contains(w.Body.String(), want) {
			t.Fatalf("stream missing %q: %s", want, w.Body.String())
		}
	}
}

func TestResponses_TranslatesCodexFunctionItems(t *testing.T) {
	var got types.ChatCompletionRequest
	h := captureRequestHandler(t, nil, &got)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
  "model":"gpt-cheap",
  "input":[
    {"type":"message","role":"user","content":[{"type":"input_text","text":"inspect files"}]},
    {"type":"function_call","call_id":"call_1","name":"shell","arguments":"{\"cmd\":\"pwd\"}"},
    {"type":"function_call_output","call_id":"call_1","output":"/workspace"}
  ],
  "tools":[{"type":"function","name":"shell","description":"Run a command","parameters":{"type":"object"}}]
}`))
	h.Responses(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", w.Code, w.Body.String())
	}
	if len(got.Messages) != 3 {
		t.Fatalf("messages = %+v", got.Messages)
	}
	if got.Messages[1].Role != "assistant" || len(got.Messages[1].ToolCalls) != 1 || got.Messages[1].ToolCalls[0].ID != "call_1" {
		t.Fatalf("function call was not translated: %+v", got.Messages[1])
	}
	if got.Messages[2].Role != "tool" || got.Messages[2].ToolCallID != "call_1" || got.Messages[2].ContentString() != "/workspace" {
		t.Fatalf("function output was not translated: %+v", got.Messages[2])
	}
	if len(got.Tools) != 1 || got.Tools[0].Function.Name != "shell" {
		t.Fatalf("tools = %+v", got.Tools)
	}
}
