package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
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

var errIOEOF = io.EOF

// scriptedStream replays a fixed list of chunks, then optionally returns a
// non-EOF error (to simulate an upstream mid-stream failure) or io.EOF.
type scriptedStream struct {
	chunks []*types.ChatCompletionChunk
	pos    int
	failAt int // index at which to return errUpstream instead of a chunk; -1 = clean EOF
}

var errUpstream = errors.New("upstream boom")

func (s *scriptedStream) ReadChunk() (*types.ChatCompletionChunk, error) {
	if s.failAt >= 0 && s.pos == s.failAt {
		return nil, errUpstream
	}
	if s.pos >= len(s.chunks) {
		return nil, errorEOF()
	}
	c := s.chunks[s.pos]
	s.pos++
	return c, nil
}
func (s *scriptedStream) Close() error { return nil }

// errorEOF returns io.EOF via the provider contract.
func errorEOF() error { return errIOEOF }

// streamProvider serves a scripted stream on SendStream.
type streamProvider struct {
	name   string
	chunks []*types.ChatCompletionChunk
	failAt int
}

func (p *streamProvider) Name() string { return p.name }
func (p *streamProvider) Send(context.Context, *types.ChatCompletionRequest, string) (*provider.Response, error) {
	return &provider.Response{StatusCode: http.StatusOK, ChatResponse: &types.ChatCompletionResponse{}}, nil
}
func (p *streamProvider) SendStream(context.Context, *types.ChatCompletionRequest, string) (provider.StreamReader, error) {
	return &scriptedStream{chunks: p.chunks, failAt: p.failAt}, nil
}

func streamHandler(t *testing.T, providerName string, chunks []*types.ChatCompletionChunk, failAt int) *Handler {
	t.Helper()
	cfg := &config.Config{}
	switch providerName {
	case "openai":
		cfg.Providers.OpenAI = &config.ProviderConfig{Models: []config.ModelConfig{{ID: "gpt-cheap", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}}}
	case "bedrock":
		cfg.Providers.Bedrock = &config.ProviderConfig{Models: []config.ModelConfig{{ID: "claude-strong", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}}}
	}
	reg := provider.NewRegistry()
	reg.Register(&streamProvider{name: providerName, chunks: chunks, failAt: failAt})
	return &Handler{router: router.NewRouter(cfg, nil), registry: reg, pricing: pricing.NewTable(cfg.Providers)}
}

// --- chunk builders ---------------------------------------------------------

func strp(s string) *string { return &s }
func intp(i int) *int       { return &i }

// textChunk is a content-only chunk on choice 0.
func textChunk(text string) *types.ChatCompletionChunk {
	return &types.ChatCompletionChunk{Object: "chat.completion.chunk", Model: "gpt-cheap",
		Choices: []types.ChunkChoice{{Index: 0, Delta: types.ChunkDelta{Content: strp(text)}}}}
}

// toolChunk carries a tool-call fragment on the given choice/tool index.
func toolChunk(choiceIdx, toolIdx int, id, name, args string) *types.ChatCompletionChunk {
	return &types.ChatCompletionChunk{Object: "chat.completion.chunk", Model: "gpt-cheap",
		Choices: []types.ChunkChoice{{Index: choiceIdx, Delta: types.ChunkDelta{ToolCalls: []types.ToolCall{{
			Index: intp(toolIdx), ID: id, Type: "function", Function: types.FunctionCall{Name: name, Arguments: args},
		}}}}}}
}

// finishChunk is a finish-only chunk (no tool delta), like the real terminal
// tool_calls chunk that must NOT race ahead of the tool deltas.
func finishChunk(reason string) *types.ChatCompletionChunk {
	return &types.ChatCompletionChunk{Object: "chat.completion.chunk", Model: "gpt-cheap",
		Choices: []types.ChunkChoice{{Index: 0, FinishReason: strp(reason)}}}
}

// doStream drives ChatCompletion with stream:true and returns the raw SSE body.
func doStream(t *testing.T, h *Handler) string {
	t.Helper()
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	return w.Body.String()
}

// sseChunks parses the data: lines of an SSE body into decoded chat chunks (skips
// [DONE]).
func sseChunks(t *testing.T, body string) []types.ChatCompletionChunk {
	t.Helper()
	var out []types.ChatCompletionChunk
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		var c types.ChatCompletionChunk
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("bad SSE chunk %q: %v", payload, err)
		}
		out = append(out, c)
	}
	return out
}

func allowAll(in types.ResponseActionInput) types.ResponseActionDecision {
	decs := make([]types.ResponseActionCallDecision, len(in.ToolCalls))
	for i := range in.ToolCalls {
		decs[i] = types.ResponseActionCallDecision{Index: i, Verdict: types.ActionAllow}
	}
	return types.ResponseActionDecision{Decisions: decs}
}

// F2: allowed streams preserve original chunk order, including a text chunk that
// arrives AFTER the first tool fragment and the terminal finish chunk.
func TestStream_OpenAIChat_AllowPreservesOrder(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		textChunk("thinking "),                   // 0: pre-tool text (flushed live)
		toolChunk(0, 0, "c1", "search", `{"q":`), // 1: first tool fragment -> buffering starts
		textChunk("mid "),                        // 2: interleaved text (must stay ordered)
		toolChunk(0, 0, "", "", `"x"}`),          // 3: rest of args
		finishChunk("tool_calls"),                // 4: terminal finish (must not race ahead)
	}
	h := streamHandler(t, "openai", chunks, -1)
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: allowAll})
	body := doStream(t, h)
	got := sseChunks(t, body)
	// Reconstruct the visible order: pre-tool text, then the buffered tail replayed
	// verbatim (tool frag, mid text, tool frag, finish).
	var seq []string
	for _, c := range got {
		for _, ch := range c.Choices {
			if ch.Delta.Content != nil {
				seq = append(seq, "text:"+*ch.Delta.Content)
			}
			if len(ch.Delta.ToolCalls) > 0 {
				seq = append(seq, "tool")
			}
			if ch.FinishReason != nil {
				seq = append(seq, "finish:"+*ch.FinishReason)
			}
		}
	}
	want := []string{"text:thinking ", "tool", "text:mid ", "tool", "finish:tool_calls"}
	if strings.Join(seq, "|") != strings.Join(want, "|") {
		t.Fatalf("order not preserved:\n got %v\nwant %v", seq, want)
	}
}

// F2: a block decision releases NO executable tool fragment and emits one ordinary
// completion carrying the envelope.
func TestStream_OpenAIChat_BlockReleasesNoToolAndEmitsEnvelope(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		toolChunk(0, 0, "c1", "rm_rf", `{"path":"/"}`),
		finishChunk("tool_calls"),
	}
	h := streamHandler(t, "openai", chunks, -1)
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(types.ResponseActionInput) types.ResponseActionDecision {
		return types.ResponseActionDecision{ReasonCode: "destructive", Decisions: []types.ResponseActionCallDecision{{Index: 0, Verdict: types.ActionBlock}}}
	}})
	body := doStream(t, h)
	// The envelope may name the blocked tool, but raw ARGUMENTS must never leak.
	if strings.Contains(body, `{\"path\":\"/\"}`) {
		t.Fatalf("blocked raw arguments leaked to client:\n%s", body)
	}
	if !strings.Contains(body, `aion_action`) || !strings.Contains(body, `block`) {
		t.Fatalf("envelope missing: %s", body)
	}
	got := sseChunks(t, body)
	for _, c := range got {
		for _, ch := range c.Choices {
			if len(ch.Delta.ToolCalls) > 0 {
				t.Fatalf("no executable tool-call fragment may be released on block: %s", body)
			}
		}
	}
}

// F7 (streaming): mixed block+hold top action is block.
func TestStream_OpenAIChat_MixedBlockHoldTopIsBlock(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		toolChunk(0, 0, "c1", "danger", `{"a":1}`),
		toolChunk(0, 1, "c2", "review", `{"b":2}`),
		finishChunk("tool_calls"),
	}
	h := streamHandler(t, "openai", chunks, -1)
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(types.ResponseActionInput) types.ResponseActionDecision {
		return types.ResponseActionDecision{Decisions: []types.ResponseActionCallDecision{
			{Index: 0, Verdict: types.ActionBlock}, {Index: 1, Verdict: types.ActionHold, HoldID: "h1"},
		}}
	}})
	body := doStream(t, h)
	if !strings.Contains(body, `\"action\":\"block\"`) {
		t.Fatalf("streaming mixed block+hold top action must be block: %s", body)
	}
}

// F4: two choices both use tool index 0 and receive DIFFERENT decisions; the hook
// must see two distinct calls (not one merged call), and a block on either fails
// the whole plan closed.
func TestStream_OpenAIChat_TwoChoicesSameToolIndexDistinct(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		toolChunk(0, 0, "c0", "alpha", `{"n":0}`),
		toolChunk(1, 0, "c1", "beta", `{"n":1}`),
		finishChunk("tool_calls"),
	}
	h := streamHandler(t, "openai", chunks, -1)
	var sawNames []string
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
		for _, c := range in.ToolCalls {
			sawNames = append(sawNames, c.Name)
		}
		// Allow choice 0's call, block choice 1's call.
		decs := make([]types.ResponseActionCallDecision, len(in.ToolCalls))
		for i, c := range in.ToolCalls {
			v := types.ActionAllow
			if c.ChoiceIndex == 1 {
				v = types.ActionBlock
			}
			decs[i] = types.ResponseActionCallDecision{Index: i, Verdict: v}
		}
		return types.ResponseActionDecision{Decisions: decs}
	}})
	body := doStream(t, h)
	if len(sawNames) != 2 {
		t.Fatalf("hook must see TWO distinct calls (not merged), saw %v", sawNames)
	}
	if !(contains(sawNames, "alpha") && contains(sawNames, "beta")) {
		t.Fatalf("both calls must be distinct, saw %v", sawNames)
	}
	// One blocked -> all-or-nothing -> no executable tool fragment released, and no
	// raw arguments leak (the envelope may name the tools + carry digests).
	if strings.Contains(body, `{\"n\":0}`) || strings.Contains(body, `{\"n\":1}`) {
		t.Fatalf("raw arguments leaked when one choice is blocked:\n%s", body)
	}
	for _, c := range sseChunks(t, body) {
		for _, ch := range c.Choices {
			if len(ch.Delta.ToolCalls) > 0 {
				t.Fatalf("no executable tool fragment may be released when one choice is blocked:\n%s", body)
			}
		}
	}
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// F5: a non-EOF read error after tool fragments must fail closed with no replay.
// Covers error after the name, mid-arguments, and after complete-looking JSON.
func TestStream_OpenAIChat_UpstreamErrorFailsClosed(t *testing.T) {
	cases := []struct {
		name   string
		chunks []*types.ChatCompletionChunk
		failAt int
	}{
		{"after name", []*types.ChatCompletionChunk{
			toolChunk(0, 0, "c1", "search", ""),
		}, 1},
		{"mid arguments", []*types.ChatCompletionChunk{
			toolChunk(0, 0, "c1", "search", `{"q":`),
		}, 1},
		{"after complete-looking json", []*types.ChatCompletionChunk{
			toolChunk(0, 0, "c1", "search", `{"q":"x"}`),
		}, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := streamHandler(t, "openai", c.chunks, c.failAt)
			called := false
			h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(types.ResponseActionInput) types.ResponseActionDecision {
				called = true
				return allowAll(types.ResponseActionInput{})
			}})
			body := doStream(t, h)
			if called {
				t.Fatalf("hook must NOT be called after an upstream stream error (%s)", c.name)
			}
			if strings.Contains(body, "search") {
				t.Fatalf("partial tool call leaked after error (%s):\n%s", c.name, body)
			}
			if !strings.Contains(body, "upstream_stream_error") {
				t.Fatalf("must fail closed with upstream_stream_error (%s):\n%s", c.name, body)
			}
		})
	}
}

// F3: keep sending data AFTER the aggregate arg ceiling is exceeded and assert the
// retained state stays bounded and the response fails closed. This drives the
// buffer directly so it can push well past the ceiling deterministically.
func TestStreamToolBuffer_BoundedAfterOverflow(t *testing.T) {
	b := newStreamToolBuffer()
	big := strings.Repeat("a", 300*1024) // one call over the per-call ceiling
	b.add(toolChunk(0, 0, "c1", "f", big))
	if !b.overflow {
		t.Fatal("per-call ceiling must trip overflow")
	}
	retainedChunks := len(b.bufferedChunks)
	// Keep sending a lot more data after overflow.
	for i := 0; i < 100; i++ {
		b.add(toolChunk(0, 0, "c1", "f", big))
	}
	if len(b.bufferedChunks) != 0 {
		t.Fatalf("retained chunks must be dropped after overflow, got %d (was %d)", len(b.bufferedChunks), retainedChunks)
	}
	if _, err := b.proposed(); err == nil {
		t.Fatal("overflow must surface as an error (fail closed)")
	}
	if !b.hasToolCalls() {
		t.Fatal("overflow still counts as governed tool calls so the caller fails closed")
	}
}

// F3: aggregate-args ceiling trips across MANY small calls, not just one big one.
func TestStreamToolBuffer_AggregateArgsCeiling(t *testing.T) {
	b := newStreamToolBuffer()
	chunk := 200 * 1024
	// Two calls of 200KiB each is under the per-call ceiling (256KiB) but a third
	// pushes the aggregate over 1MiB... use 6 * 200KiB = 1.2MiB.
	for i := 0; i < 6; i++ {
		b.add(toolChunk(0, i, "c", "f", strings.Repeat("b", chunk)))
		if b.overflow {
			break
		}
	}
	if !b.overflow {
		t.Fatal("aggregate args ceiling must trip overflow")
	}
}

// F3: call-count ceiling trips and stops retention.
func TestStreamToolBuffer_CallCountCeiling(t *testing.T) {
	b := newStreamToolBuffer()
	for i := 0; i <= types.MaxBufferedToolCallCount+10; i++ {
		b.add(toolChunk(0, i, "c", "f", "{}"))
		if b.overflow {
			break
		}
	}
	if !b.overflow {
		t.Fatal("call-count ceiling must trip overflow")
	}
	if len(b.bufferedChunks) != 0 {
		t.Fatal("chunks dropped after overflow")
	}
}

// F6: the Responses endpoint (streaming) must report the ingress protocol
// "openai_responses" to the hook, not the native "openai_chat".
func TestStream_Responses_ReportsOpenAIResponsesProtocol(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		toolChunk(0, 0, "c1", "search", `{"q":"x"}`),
		finishChunk("tool_calls"),
	}
	h := streamHandler(t, "openai", chunks, -1)
	var sawProtocol string
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
		sawProtocol = in.Protocol
		return allowAll(in)
	}})
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"gpt-cheap","stream":true,"input":"hi"}`))
	h.Responses(w, r)
	if sawProtocol != "openai_responses" {
		t.Fatalf("Responses streaming ingress protocol = %q, want openai_responses", sawProtocol)
	}
}

// F6: the native chat endpoint still reports openai_chat.
func TestStream_OpenAIChat_ReportsOpenAIChatProtocol(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{toolChunk(0, 0, "c1", "s", `{}`), finishChunk("tool_calls")}
	h := streamHandler(t, "openai", chunks, -1)
	var sawProtocol string
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
		sawProtocol = in.Protocol
		return allowAll(in)
	}})
	doStream(t, h)
	if sawProtocol != "openai_chat" {
		t.Fatalf("chat ingress protocol = %q, want openai_chat", sawProtocol)
	}
}

// doAnthropicStream drives the Anthropic ingress with stream:true and returns the
// raw SSE body.
func doAnthropicStream(t *testing.T, h *Handler, model string) string {
	t.Helper()
	w := httptest.NewRecorder()
	body := `{"model":"` + model + `","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	h.AnthropicMessages(w, r)
	return w.Body.String()
}

// F2 (Anthropic): an allowed tool stream replays tool_use; the hook is told the
// ingress protocol "anthropic".
func TestStream_Anthropic_AllowReplaysToolUseAndReportsProtocol(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		toolChunk(0, 0, "toolu_1", "search", `{"q":`),
		toolChunk(0, 0, "", "", `"x"}`),
		finishChunk("tool_calls"),
	}
	h := streamHandler(t, "bedrock", chunks, -1)
	var sawProtocol string
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
		sawProtocol = in.Protocol
		return allowAll(in)
	}})
	body := doAnthropicStream(t, h, "claude-strong")
	if sawProtocol != "anthropic" {
		t.Fatalf("anthropic ingress protocol = %q, want anthropic", sawProtocol)
	}
	if !strings.Contains(body, `"type":"tool_use"`) {
		t.Fatalf("allowed tool call must be replayed as a tool_use block:\n%s", body)
	}
	if !strings.Contains(body, "search") {
		t.Fatalf("allowed tool name must appear:\n%s", body)
	}
}

// F2 (Anthropic): a held tool call releases NO tool_use; it emits the envelope as
// a text block.
func TestStream_Anthropic_HoldEmitsEnvelopeNoToolUse(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		toolChunk(0, 0, "toolu_1", "wire_money", `{"amt":999}`),
		finishChunk("tool_calls"),
	}
	h := streamHandler(t, "bedrock", chunks, -1)
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(types.ResponseActionInput) types.ResponseActionDecision {
		return types.ResponseActionDecision{ReasonCode: "approval", Decisions: []types.ResponseActionCallDecision{{Index: 0, Verdict: types.ActionHold, HoldID: "h-7"}}}
	}})
	body := doAnthropicStream(t, h, "claude-strong")
	if strings.Contains(body, `"type":"tool_use"`) {
		t.Fatalf("held tool call must NOT be released as tool_use:\n%s", body)
	}
	if strings.Contains(body, `{\"amt\":999}`) {
		t.Fatalf("raw held arguments leaked:\n%s", body)
	}
	if !strings.Contains(body, "aion_action") || !strings.Contains(body, "hold") {
		t.Fatalf("held envelope missing:\n%s", body)
	}
}

// F5 (Anthropic): a mid-stream error after a tool fragment fails closed with no
// replay.
func TestStream_Anthropic_UpstreamErrorFailsClosed(t *testing.T) {
	chunks := []*types.ChatCompletionChunk{
		toolChunk(0, 0, "toolu_1", "search", `{"q":`),
	}
	h := streamHandler(t, "bedrock", chunks, 1)
	called := false
	h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
		called = true
		return allowAll(in)
	}})
	body := doAnthropicStream(t, h, "claude-strong")
	if called {
		t.Fatal("hook must NOT run after an upstream stream error")
	}
	if strings.Contains(body, `"type":"tool_use"`) || strings.Contains(body, "search") {
		t.Fatalf("partial tool call leaked after error:\n%s", body)
	}
	if !strings.Contains(body, "upstream_stream_error") {
		t.Fatalf("must fail closed with upstream_stream_error:\n%s", body)
	}
}
