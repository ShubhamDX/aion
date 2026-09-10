package proxy

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ShubhamDX/aion/internal/types"
)

func prefixChunk(text, finish string) *types.ChatCompletionChunk {
	c := types.ChunkChoice{Index: 0}
	if text != "" {
		c.Delta.Content = &text
	}
	if finish != "" {
		c.FinishReason = &finish
	}
	return &types.ChatCompletionChunk{Choices: []types.ChunkChoice{c}}
}

func TestStreamPrefixCompletedText(t *testing.T) {
	req := &types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}}}
	var p streamPrefix
	p.add(prefixChunk("hello ", ""))
	p.add(prefixChunk("Ω", "stop"))
	next := &types.ChatCompletionRequest{Messages: append(append([]types.Message{}, req.Messages...), types.Message{Role: "assistant", Content: json.RawMessage(`"hello \u03a9"`)}, types.Message{Role: "user", Content: json.RawMessage(`"continue"`)})}
	want := types.SessionMaterialFromRequest(next, "session").CachePrefixMaterialSHA256
	if got := p.digest(req, true); got == "" || got != want {
		t.Fatalf("prefix mismatch: %q != %q", got, want)
	}
	if p.digest(req, false) != "" {
		t.Fatal("partial response published warmth")
	}
}

func TestStreamPrefixRejectsIncompleteOrUnboundedResponse(t *testing.T) {
	for name, chunks := range map[string][]*types.ChatCompletionChunk{
		"missing finish":   {prefixChunk("hello", "")},
		"truncated":        {prefixChunk("hello", "length")},
		"overflow":         {prefixChunk(strings.Repeat("a", maxStreamPrefixText+1), "stop")},
		"after finish":     {prefixChunk("hello", "stop"), prefixChunk("extra", "")},
		"multiple choices": {{Choices: []types.ChunkChoice{{Index: 1}}}, prefixChunk("hello", "stop")},
	} {
		t.Run(name, func(t *testing.T) {
			var p streamPrefix
			for _, c := range chunks {
				p.add(c)
			}
			if p.digest(&types.ChatCompletionRequest{}, true) != "" {
				t.Fatal("unsafe warmth published")
			}
		})
	}
}

type brokenStreamWriter struct{ *httptest.ResponseRecorder }

func (w brokenStreamWriter) Write(b []byte) (int, error) { return 0, errors.New("client disconnected") }

func TestStreamDeliveryWriterTracksFailure(t *testing.T) {
	w := &streamDeliveryWriter{ResponseWriter: brokenStreamWriter{httptest.NewRecorder()}}
	_, _ = w.Write([]byte("data"))
	if !w.failed {
		t.Fatal("write failure lost")
	}
}

func TestStreamPrefixToolIdentity(t *testing.T) {
	idx := 0
	finish := "tool_calls"
	var p streamPrefix
	p.add(&types.ChatCompletionChunk{Choices: []types.ChunkChoice{{Delta: types.ChunkDelta{ToolCalls: []types.ToolCall{{Index: &idx, ID: "call-1", Type: "function", Function: types.FunctionCall{Name: "read", Arguments: `{"id":`}}}}}}})
	p.add(&types.ChatCompletionChunk{Choices: []types.ChunkChoice{{Delta: types.ChunkDelta{ToolCalls: []types.ToolCall{{Index: &idx, Function: types.FunctionCall{Arguments: `1}`}}}}, FinishReason: &finish}}})
	req := &types.ChatCompletionRequest{}
	want := types.NextCachePrefixMaterial(req, &types.ChatCompletionResponse{Choices: []types.Choice{{Message: types.Message{Role: "assistant", Content: json.RawMessage(`null`), ToolCalls: []types.ToolCall{{ID: "call-1", Type: "function", Function: types.FunctionCall{Name: "read", Arguments: `{"id":1}`}}}}}}})
	if got := p.digest(req, true); got == "" || got != want {
		t.Fatalf("tool prefix mismatch: %q != %q", got, want)
	}
	p.invalidate()
	if p.digest(req, true) != "" {
		t.Fatal("blocked tool response published warmth")
	}
}

func TestStreamingIngressPublishesOnlyDeliveredCompleteWarmth(t *testing.T) {
	for _, protocol := range []string{"openai", "anthropic"} {
		for _, scenario := range []string{"complete", "read-error", "write-error", "overflow"} {
			t.Run(protocol+"/"+scenario, func(t *testing.T) {
				text := "hello Ω"
				if scenario == "overflow" {
					text = strings.Repeat("x", maxStreamPrefixText+1)
				}
				chunks := []*types.ChatCompletionChunk{prefixChunk(text, "stop")}
				fail := -1
				if scenario == "read-error" {
					fail = 1
				}
				providerName, model := "openai", "gpt-cheap"
				if protocol == "anthropic" {
					providerName, model = "bedrock", "claude-strong"
				}
				h := streamHandler(t, providerName, chunks, fail)
				seen := make(chan types.PostResponseInput, 1)
				h.SetGatewayHooks(&types.GatewayHooks{PostResponse: func(in types.PostResponseInput) { seen <- in }})
				rec := httptest.NewRecorder()
				var w http.ResponseWriter = rec
				if scenario == "write-error" {
					w = brokenStreamWriter{rec}
				}
				body := `{"model":"` + model + `","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`
				r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
				if protocol == "anthropic" {
					h.AnthropicMessages(w, r)
				} else {
					h.ChatCompletion(w, r)
				}
				select {
				case in := <-seen:
					got := in.SessionMaterial.NextCachePrefixMaterialSHA256
					if (got != "") != (scenario == "complete") {
						t.Fatalf("unexpected warmth %q", got)
					}
					if scenario == "complete" && !strings.Contains(rec.Body.String(), text) {
						t.Fatal("warm response not delivered")
					}
				case <-time.After(2 * time.Second):
					t.Fatal("post-response hook missing")
				}
			})
		}
	}
}

func TestOutputControlBlockStopsEveryIngress(t *testing.T) {
	for _, protocol := range []string{"openai", "anthropic"} {
		for _, stream := range []bool{false, true} {
			h := stubHandler("must not be released")
			h.SetGatewayHooks(&types.GatewayHooks{ApplyOutputControl: func(types.OutputControlInput) *types.OutputControlResult {
				return &types.OutputControlResult{Block: true}
			}})
			body, _ := json.Marshal(map[string]any{"model": "haiku", "stream": stream, "max_tokens": 64, "messages": []map[string]string{{"role": "user", "content": "hello"}}})
			w := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(string(body)))
			if protocol == "anthropic" {
				h.AnthropicMessages(w, r)
			} else {
				h.ChatCompletion(w, r)
			}
			if w.Code != http.StatusServiceUnavailable || strings.Contains(w.Body.String(), "must not be released") {
				t.Fatalf("protocol=%s stream=%v status=%d body=%s", protocol, stream, w.Code, w.Body.String())
			}
		}
	}
}
