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

// dec is a shorthand for a call decision.
func dec(idx int, v ResponseActionVerdict) ResponseActionCallDecision {
	return ResponseActionCallDecision{Index: idx, Verdict: v}
}

// TestResponseActionDecision_Validate covers the complete-mapping contract: the
// decision must be exactly one in-range, unique, known-verdict decision per
// proposed call. Every malformed shape must fail closed (Validate=false).
func TestResponseActionDecision_Validate(t *testing.T) {
	cases := []struct {
		name     string
		proposed int
		decs     []ResponseActionCallDecision
		want     bool
	}{
		{"empty decision over one call (missing) -> invalid",
			1, nil, false},
		{"empty over zero calls -> valid",
			0, nil, true},
		{"exact one-to-one allow -> valid",
			2, []ResponseActionCallDecision{dec(0, ActionAllow), dec(1, ActionBlock)}, true},
		{"too few decisions -> invalid",
			2, []ResponseActionCallDecision{dec(0, ActionAllow)}, false},
		{"too many decisions -> invalid",
			1, []ResponseActionCallDecision{dec(0, ActionAllow), dec(1, ActionAllow)}, false},
		{"duplicate index -> invalid",
			2, []ResponseActionCallDecision{dec(0, ActionAllow), dec(0, ActionBlock)}, false},
		{"negative index -> invalid",
			2, []ResponseActionCallDecision{dec(-1, ActionAllow), dec(1, ActionAllow)}, false},
		{"out-of-range index -> invalid",
			2, []ResponseActionCallDecision{dec(0, ActionAllow), dec(2, ActionAllow)}, false},
		{"unknown verdict -> invalid",
			1, []ResponseActionCallDecision{{Index: 0, Verdict: ResponseActionVerdict(99)}}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := ResponseActionDecision{Decisions: c.decs}
			if got := d.Validate(c.proposed); got != c.want {
				t.Fatalf("Validate(%d) = %v, want %v", c.proposed, got, c.want)
			}
		})
	}
}

// TestResponseActionDecision_AllAllowedValidated: a malformed decision never
// reports all-allowed (fail closed), and a valid all-allow does.
func TestResponseActionDecision_AllAllowedValidated(t *testing.T) {
	// One allow decision against a TWO-call response must NOT release (missing
	// decision for the second call).
	one := ResponseActionDecision{Decisions: []ResponseActionCallDecision{dec(0, ActionAllow)}}
	if one.AllAllowedValidated(2) {
		t.Fatal("one allow over two calls must fail closed")
	}
	// A complete all-allow releases.
	full := ResponseActionDecision{Decisions: []ResponseActionCallDecision{dec(0, ActionAllow), dec(1, ActionAllow)}}
	if !full.AllAllowedValidated(2) {
		t.Fatal("complete all-allow must release")
	}
	// Empty decision over zero calls is vacuously all-allowed (but the caller only
	// invokes the hook when there is >=1 call).
	if !(ResponseActionDecision{}).AllAllowedValidated(0) {
		t.Fatal("empty over zero calls should be all-allowed")
	}
}

// TestResponseActionDecision_TopSeverityAction: block outranks hold outranks
// allow, and a malformed decision fails closed to block.
func TestResponseActionDecision_TopSeverityAction(t *testing.T) {
	blockPlusHold := ResponseActionDecision{Decisions: []ResponseActionCallDecision{
		dec(0, ActionBlock), dec(1, ActionHold),
	}}
	if got := blockPlusHold.TopSeverityAction(2); got != "block" {
		t.Fatalf("block+hold top action = %q, want block", got)
	}
	holdPlusAllow := ResponseActionDecision{Decisions: []ResponseActionCallDecision{
		dec(0, ActionHold), dec(1, ActionAllow),
	}}
	if got := holdPlusAllow.TopSeverityAction(2); got != "hold" {
		t.Fatalf("hold+allow top action = %q, want hold", got)
	}
	// Malformed -> block.
	if got := (ResponseActionDecision{}).TopSeverityAction(2); got != "block" {
		t.Fatalf("malformed top action = %q, want block", got)
	}
}
