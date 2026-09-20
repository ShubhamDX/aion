package provider

import (
	"context"
	"github.com/ShubhamDX/aion/internal/types"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseBedrockResponseOpenAIStyle(t *testing.T) {
	body := []byte(`{"choices":[{"finish_reason":"stop","message":{"content":"hello from qwen","role":"assistant"}}],"usage":{"completion_tokens":7,"prompt_tokens":11,"total_tokens":18}}`)

	resp, err := parseBedrockResponse(body, "qwen.qwen3-32b-v1:0")
	if err != nil {
		t.Fatalf("parseBedrockResponse: %v", err)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(resp.Choices))
	}
	if got := resp.Choices[0].Message.ContentString(); got != "hello from qwen" {
		t.Fatalf("content = %q, want %q", got, "hello from qwen")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "stop")
	}
	if resp.Usage.PromptTokens != 11 || resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Fatalf("usage = %+v, want prompt=11 completion=7 total=18", resp.Usage)
	}
	if resp.Model != "qwen.qwen3-32b-v1:0" {
		t.Fatalf("model = %q, want %q", resp.Model, "qwen.qwen3-32b-v1:0")
	}
}

func TestParseBedrockResponseAnthropicStyle(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello from claude"}],"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":4}}`)

	resp, err := parseBedrockResponse(body, "anthropic.claude-3-sonnet")
	if err != nil {
		t.Fatalf("parseBedrockResponse: %v", err)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(resp.Choices))
	}
	if got := resp.Choices[0].Message.ContentString(); got != "hello from claude" {
		t.Fatalf("content = %q, want %q", got, "hello from claude")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "stop")
	}
	if resp.Usage.PromptTokens != 9 || resp.Usage.CompletionTokens != 4 || resp.Usage.TotalTokens != 13 {
		t.Fatalf("usage = %+v, want prompt=9 completion=4 total=13", resp.Usage)
	}
}

func TestParseBedrockResponseUnrecognizedShape(t *testing.T) {
	body := []byte(`{"foo":"bar"}`)

	if _, err := parseBedrockResponse(body, "some-model"); err == nil {
		t.Fatal("expected an error for a response with neither content nor choices, got nil")
	}
}

func TestParseBedrockResponseRejectsInvalidEnvelopes(t *testing.T) {
	for _, body := range []string{
		`{"choices":null}`, `{"choices":[]}`, `{"content":null}`,
		`{"choices":[{"message":{"content":"x"}}],"content":[]}`,
	} {
		t.Run(body, func(t *testing.T) {
			if _, err := parseBedrockResponse([]byte(body), "model"); err == nil {
				t.Fatal("invalid envelope accepted")
			}
		})
	}
}

func TestParseBedrockResponseErrorDoesNotExposeBody(t *testing.T) {
	_, err := parseBedrockResponse([]byte(`{"provider_detail":"private-customer-content"}`), "model")
	if err == nil || strings.Contains(err.Error(), "private-customer-content") {
		t.Fatalf("unsafe error: %v", err)
	}
}

func TestParseBedrockResponseAllowsEmptyAnthropicRefusal(t *testing.T) {
	resp, err := parseBedrockResponse([]byte(`{"content":[],"stop_reason":"refusal","usage":{"input_tokens":9,"output_tokens":0}}`), "model")
	if err != nil || len(resp.Choices) != 1 || resp.Usage.PromptTokens != 9 {
		t.Fatalf("refusal lost: %+v %v", resp, err)
	}
}

func TestBedrockSendDecodesInvokeEnvelope(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		wantError  bool
	}{
		{"openai", `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10}}`, false},
		{"anthropic", `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":1}}`, false},
		{"malformed", `{"choices":null}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/model/fixture-model/invoke" || r.Header.Get("Authorization") != "Bearer test-only" {
					t.Error("unexpected invoke request")
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			}))
			defer upstream.Close()
			p := &BedrockProvider{baseURL: upstream.URL, client: upstream.Client(), bearerToken: "test-only"}
			resp, err := p.Send(context.Background(), &types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: "test"}}}, "fixture-model")
			if tc.wantError {
				if err == nil {
					t.Fatal("malformed envelope accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if resp.Model != "fixture-model" || len(resp.Choices) != 1 || resp.Choices[0].Message.ContentString() != "ok" || resp.Usage.PromptTokens != 9 || resp.Usage.CompletionTokens != 1 {
				t.Fatalf("lost content or usage: %+v", resp)
			}
		})
	}
}
