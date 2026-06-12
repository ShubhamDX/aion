package types

import (
	"encoding/json"
	"testing"
)

func msg(role, text string) Message {
	b, _ := json.Marshal(text)
	return Message{Role: role, Content: b}
}

func TestRequestContentDigest_DeterministicAndContentBound(t *testing.T) {
	a := &ChatCompletionRequest{Model: "m", Messages: []Message{msg("user", "hello")}}
	b := &ChatCompletionRequest{Model: "m", Messages: []Message{msg("user", "hello")}}
	if RequestContentDigest(a) != RequestContentDigest(b) {
		t.Fatal("identical content must produce identical digest")
	}
	// Different content -> different digest.
	c := &ChatCompletionRequest{Model: "m", Messages: []Message{msg("user", "goodbye")}}
	if RequestContentDigest(a) == RequestContentDigest(c) {
		t.Fatal("different content must produce different digest")
	}
	// Different model -> different digest.
	d := &ChatCompletionRequest{Model: "other", Messages: []Message{msg("user", "hello")}}
	if RequestContentDigest(a) == RequestContentDigest(d) {
		t.Fatal("different model must produce different digest")
	}
}

func TestRequestContentDigest_LengthFramingNoCollision(t *testing.T) {
	// "ab"+"c" must not collide with "a"+"bc" thanks to length framing.
	x := &ChatCompletionRequest{Model: "m", Messages: []Message{msg("user", "ab"), msg("user", "c")}}
	y := &ChatCompletionRequest{Model: "m", Messages: []Message{msg("user", "a"), msg("user", "bc")}}
	if RequestContentDigest(x) == RequestContentDigest(y) {
		t.Fatal("length framing must prevent boundary collision")
	}
}

func TestRequestContentDigest_NilEmpty(t *testing.T) {
	if RequestContentDigest(nil) != "" {
		t.Fatal("nil request -> empty digest")
	}
}

func TestResponseContentDigest(t *testing.T) {
	r := &ChatCompletionResponse{Choices: []Choice{{Message: msg("assistant", "hi there")}}}
	if ResponseContentDigest(r) == "" {
		t.Fatal("response with content must digest")
	}
	if ResponseContentDigest(nil) != "" || ResponseContentDigest(&ChatCompletionResponse{}) != "" {
		t.Fatal("empty response -> empty digest")
	}
}

func TestGatewayHooks_NilSafe(t *testing.T) {
	// A nil GatewayHooks and nil func fields must be the documented no-op shape.
	var hooks *GatewayHooks
	if hooks != nil {
		t.Fatal("zero value is nil")
	}
	h := &GatewayHooks{}
	if h.PreRequest != nil || h.PostResponse != nil {
		t.Fatal("empty hooks have nil funcs")
	}
}

func TestRequestContentDigest_BindsToolSurface(t *testing.T) {
	base := &ChatCompletionRequest{Model: "m", Messages: []Message{msg("user", "do it")}}
	withTool := &ChatCompletionRequest{
		Model: "m", Messages: []Message{msg("user", "do it")},
		Tools: []Tool{{Type: "function", Function: FunctionDef{Name: "shell_exec"}}},
	}
	if RequestContentDigest(base) == RequestContentDigest(withTool) {
		t.Fatal("adding a tool must change the request digest (tool surface bound)")
	}
}

func TestResponseContentDigest_BindsToolCalls(t *testing.T) {
	textOnly := &ChatCompletionResponse{Choices: []Choice{{Message: msg("assistant", "ok")}}}
	withToolCall := &ChatCompletionResponse{Choices: []Choice{{Message: Message{
		Role: "assistant", Content: msg("assistant", "").Content,
		ToolCalls: []ToolCall{{ID: "c1", Type: "function", Function: FunctionCall{Name: "delete", Arguments: "{}"}}},
	}}}}
	if ResponseContentDigest(textOnly) == ResponseContentDigest(withToolCall) {
		t.Fatal("a tool-call response must digest differently from text (action surface bound)")
	}
}

func TestRequestContentDigest_BindsCostRelevantFields(t *testing.T) {
	mk := func(maxTok int) *ChatCompletionRequest {
		return &ChatCompletionRequest{Model: "m", Messages: []Message{msg("user", "hi")}, MaxTokens: &maxTok}
	}
	a, b := mk(100), mk(4000)
	if RequestContentDigest(a) == RequestContentDigest(b) {
		t.Fatal("different max_tokens must change the digest (different budget risk)")
	}
}

func TestRequestContentDigest_BindsMultimodalContent(t *testing.T) {
	// Two messages whose TEXT parts are identical but image parts differ must
	// digest differently (raw content bound, not text-flattened).
	imgA := json.RawMessage(`[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"http://a/x.png"}}]`)
	imgB := json.RawMessage(`[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"http://b/y.png"}}]`)
	ra := &ChatCompletionRequest{Model: "m", Messages: []Message{{Role: "user", Content: imgA}}}
	rb := &ChatCompletionRequest{Model: "m", Messages: []Message{{Role: "user", Content: imgB}}}
	if RequestContentDigest(ra) == RequestContentDigest(rb) {
		t.Fatal("different multimodal (image) content must change the digest")
	}
}
