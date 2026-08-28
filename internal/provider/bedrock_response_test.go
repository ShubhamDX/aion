package provider

import "testing"

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
