package proxy

import (
	"context"
	"encoding/json"
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

// toolCallProvider returns a fixed non-streaming response carrying the given
// tool calls, so the response-action hook has complete calls to evaluate.
type toolCallProvider struct{ calls []types.ToolCall }

func (p toolCallProvider) Name() string { return "openai" }
func (p toolCallProvider) Send(context.Context, *types.ChatCompletionRequest, string) (*provider.Response, error) {
	return &provider.Response{
		StatusCode: http.StatusOK,
		ChatResponse: &types.ChatCompletionResponse{
			Choices: []types.Choice{{
				Message:      types.Message{Role: "assistant", ToolCalls: p.calls},
				FinishReason: "tool_calls",
			}},
			Usage: types.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
	}, nil
}
func (p toolCallProvider) SendStream(context.Context, *types.ChatCompletionRequest, string) (provider.StreamReader, error) {
	return nil, nil
}

func toolHandler(t *testing.T, calls []types.ToolCall) *Handler {
	t.Helper()
	cfg := &config.Config{}
	cfg.Providers.OpenAI = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "gpt-cheap", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}},
	}
	reg := provider.NewRegistry()
	reg.Register(toolCallProvider{calls: calls})
	return &Handler{router: router.NewRouter(cfg, nil), registry: reg, pricing: pricing.NewTable(cfg.Providers)}
}

func tc(id, name, args string) types.ToolCall {
	return types.ToolCall{ID: id, Type: "function", Function: types.FunctionCall{Name: name, Arguments: args}}
}

func doChat(t *testing.T, h *Handler) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-cheap","messages":[{"role":"user","content":"hi"}]}`))
	h.ChatCompletion(w, r)
	return w
}

func decodeChat(t *testing.T, w *httptest.ResponseRecorder) types.ChatCompletionResponse {
	t.Helper()
	var resp types.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v\n%s", err, w.Body.String())
	}
	return resp
}

func TestResponseAction_AllowPassesToolCallsThrough(t *testing.T) {
	h := toolHandler(t, []types.ToolCall{tc("c1", "search", `{"q":"x"}`)})
	h.SetGatewayHooks(&types.GatewayHooks{
		ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
			return types.ResponseActionDecision{Decisions: []types.ResponseActionCallDecision{{Index: 0, Verdict: types.ActionAllow}}}
		},
	})
	resp := decodeChat(t, doChat(t, h))
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("allowed tool call must pass through, got %+v", resp.Choices[0].Message)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish reason should be unchanged, got %q", resp.Choices[0].FinishReason)
	}
}

func TestResponseAction_BlockReplacesWithEnvelopeAndNoToolCall(t *testing.T) {
	h := toolHandler(t, []types.ToolCall{tc("c1", "rm_rf", `{"path":"/"}`)})
	h.SetGatewayHooks(&types.GatewayHooks{
		ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
			// The hook must receive the digest, never see raw args leak into a tool call.
			if in.ToolCalls[0].ArgsDigest == "" {
				t.Error("hook must receive an args digest")
			}
			return types.ResponseActionDecision{
				ReasonCode: "destructive",
				Decisions:  []types.ResponseActionCallDecision{{Index: 0, Verdict: types.ActionBlock, PolicyClass: "destructive"}},
			}
		},
	})
	resp := decodeChat(t, doChat(t, h))
	if len(resp.Choices[0].Message.ToolCalls) != 0 {
		t.Fatal("blocked tool call must NOT reach the client")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("blocked response must be an ordinary completion (stop), got %q", resp.Choices[0].FinishReason)
	}
	// The content carries the action envelope with a block action and the digest.
	var content string
	if err := json.Unmarshal(resp.Choices[0].Message.Content, &content); err != nil {
		t.Fatalf("content not a JSON string: %v", err)
	}
	if !strings.Contains(content, `"aion_action"`) || !strings.Contains(content, `"action":"block"`) {
		t.Fatalf("envelope missing block action: %s", content)
	}
	// The envelope carries the digest, never the raw arguments.
	if strings.Contains(content, `{"path":"/"}`) {
		t.Fatalf("raw arguments leaked into the envelope: %s", content)
	}
}

func TestResponseAction_MultiToolAllOrNothing(t *testing.T) {
	h := toolHandler(t, []types.ToolCall{
		tc("c1", "safe", `{"a":1}`),
		tc("c2", "danger", `{"b":2}`),
	})
	h.SetGatewayHooks(&types.GatewayHooks{
		ResponseAction: func(in types.ResponseActionInput) types.ResponseActionDecision {
			if len(in.ToolCalls) != 2 {
				t.Fatalf("hook must see every call, got %d", len(in.ToolCalls))
			}
			// Allow the first, hold the second: all-or-nothing means NEITHER is released.
			return types.ResponseActionDecision{Decisions: []types.ResponseActionCallDecision{
				{Index: 0, Verdict: types.ActionAllow},
				{Index: 1, Verdict: types.ActionHold, HoldID: "hold-1"},
			}}
		},
	})
	resp := decodeChat(t, doChat(t, h))
	if len(resp.Choices[0].Message.ToolCalls) != 0 {
		t.Fatal("if any call is held, NO tool call may be released")
	}
	var content string
	_ = json.Unmarshal(resp.Choices[0].Message.Content, &content)
	if !strings.Contains(content, `"action":"hold"`) || !strings.Contains(content, "hold-1") {
		t.Fatalf("envelope must carry the hold and its id: %s", content)
	}
}

func TestResponseAction_DisabledLeavesResponseUnchanged(t *testing.T) {
	h := toolHandler(t, []types.ToolCall{tc("c1", "search", `{"q":"x"}`)})
	// No hooks installed.
	resp := decodeChat(t, doChat(t, h))
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatal("with no hook, tool calls pass through unchanged")
	}
}

// envelopeContentOf decodes the action envelope carried as the choice's content.
func envelopeContentOf(t *testing.T, resp types.ChatCompletionResponse) string {
	t.Helper()
	var content string
	if err := json.Unmarshal(resp.Choices[0].Message.Content, &content); err != nil {
		t.Fatalf("content not a JSON string: %v", err)
	}
	return content
}

// F1 (non-streaming): a decision that is not a complete mapping over the proposed
// calls must fail closed to block, never release an unclassified call.
func TestResponseAction_NonStream_MalformedDecisionFailsClosed(t *testing.T) {
	cases := []struct {
		name string
		hook func(types.ResponseActionInput) types.ResponseActionDecision
	}{
		{"empty decision", func(types.ResponseActionInput) types.ResponseActionDecision {
			return types.ResponseActionDecision{}
		}},
		{"missing one (one allow over two calls)", func(types.ResponseActionInput) types.ResponseActionDecision {
			return types.ResponseActionDecision{Decisions: []types.ResponseActionCallDecision{{Index: 0, Verdict: types.ActionAllow}}}
		}},
		{"duplicate index", func(types.ResponseActionInput) types.ResponseActionDecision {
			return types.ResponseActionDecision{Decisions: []types.ResponseActionCallDecision{
				{Index: 0, Verdict: types.ActionAllow}, {Index: 0, Verdict: types.ActionAllow},
			}}
		}},
		{"out-of-range index", func(types.ResponseActionInput) types.ResponseActionDecision {
			return types.ResponseActionDecision{Decisions: []types.ResponseActionCallDecision{
				{Index: 0, Verdict: types.ActionAllow}, {Index: 5, Verdict: types.ActionAllow},
			}}
		}},
		{"unknown verdict", func(types.ResponseActionInput) types.ResponseActionDecision {
			return types.ResponseActionDecision{Decisions: []types.ResponseActionCallDecision{
				{Index: 0, Verdict: types.ActionAllow}, {Index: 1, Verdict: types.ResponseActionVerdict(42)},
			}}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := toolHandler(t, []types.ToolCall{tc("c1", "a", `{"x":1}`), tc("c2", "b", `{"y":2}`)})
			h.SetGatewayHooks(&types.GatewayHooks{ResponseAction: c.hook})
			resp := decodeChat(t, doChat(t, h))
			if len(resp.Choices[0].Message.ToolCalls) != 0 {
				t.Fatalf("malformed decision must release NO tool call, got %d", len(resp.Choices[0].Message.ToolCalls))
			}
			if resp.Choices[0].FinishReason != "stop" {
				t.Fatalf("must be an ordinary completion, got %q", resp.Choices[0].FinishReason)
			}
			if content := envelopeContentOf(t, resp); !strings.Contains(content, `"action":"block"`) {
				t.Fatalf("malformed decision must fail closed to block: %s", content)
			}
		})
	}
}

// F7 (non-streaming): a mixed block+hold decision advertises the top action as
// block (block outranks hold), while each call keeps its own verdict.
func TestResponseAction_NonStream_MixedBlockHoldTopIsBlock(t *testing.T) {
	h := toolHandler(t, []types.ToolCall{tc("c1", "danger", `{"a":1}`), tc("c2", "review", `{"b":2}`)})
	h.SetGatewayHooks(&types.GatewayHooks{
		ResponseAction: func(types.ResponseActionInput) types.ResponseActionDecision {
			return types.ResponseActionDecision{ReasonCode: "mixed", Decisions: []types.ResponseActionCallDecision{
				{Index: 0, Verdict: types.ActionBlock, PolicyClass: "destructive"},
				{Index: 1, Verdict: types.ActionHold, HoldID: "h-9"},
			}}
		},
	})
	resp := decodeChat(t, doChat(t, h))
	if len(resp.Choices[0].Message.ToolCalls) != 0 {
		t.Fatal("no tool call may be released")
	}
	content := envelopeContentOf(t, resp)
	if !strings.Contains(content, `"aion_action":{"action":"block"`) {
		t.Fatalf("mixed block+hold top action must be block: %s", content)
	}
	// Per-call verdicts retained.
	if !strings.Contains(content, `"action":"block"`) || !strings.Contains(content, `"action":"hold"`) {
		t.Fatalf("per-call block and hold must both appear: %s", content)
	}
	if !strings.Contains(content, "h-9") {
		t.Fatalf("hold id must be carried: %s", content)
	}
}
