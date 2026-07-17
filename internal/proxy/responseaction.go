package proxy

import (
	"encoding/json"
	"errors"
	"sort"

	"github.com/ShubhamDX/aion/internal/types"
)

// errToolArgsTooLarge is returned when buffered streaming tool-call arguments
// exceed the documented ceiling; the caller fails closed (blocks).
var errToolArgsTooLarge = errors.New("proxy: buffered tool-call arguments exceed limit")

// Response-action governance (PR 6). When a ResponseAction hook is installed the
// proxy normalizes every complete model-proposed tool call in a response and
// asks the hook for an allow/block/hold decision per call BEFORE releasing any
// tool call to the client. If every call is allowed the response is released
// unchanged; otherwise NO tool call is released and the response is rewritten to
// a protocol-valid completion carrying the action envelope (all-or-nothing, so a
// client that auto-executes has nothing to execute). OSS owns the seam and the
// envelope shape; policy lives in the embedding product.

// aionActionEnvelope is the one fixed, machine-parseable shape a governed
// response carries in place of a held or blocked tool call, across every ingress
// protocol. It is emitted as ordinary assistant content (never a synthesized
// tool call) and carries the arguments DIGEST only, never raw arguments.
type aionActionEnvelope struct {
	AIONAction aionActionBody `json:"aion_action"`
}

type aionActionBody struct {
	Action     string                `json:"action"` // "hold" | "block"
	ReasonCode string                `json:"reason_code,omitempty"`
	Calls      []aionActionCallEntry `json:"calls"`
}

type aionActionCallEntry struct {
	ToolName    string `json:"tool_name"`
	CallID      string `json:"call_id,omitempty"`
	ArgsDigest  string `json:"args_digest"`
	Action      string `json:"action"` // "allow" | "block" | "hold"
	HoldID      string `json:"hold_id,omitempty"`
	PolicyClass string `json:"policy_class,omitempty"`
}

// proposedToolCallsFromResponse extracts every complete tool call across all
// choices of a non-streaming response, in stable index order.
func proposedToolCallsFromResponse(resp *types.ChatCompletionResponse) []types.ProposedToolCall {
	if resp == nil {
		return nil
	}
	var out []types.ProposedToolCall
	idx := 0
	for _, choice := range resp.Choices {
		for _, tc := range choice.Message.ToolCalls {
			out = append(out, types.ProposedToolCall{
				Index:      idx,
				ID:         tc.ID,
				Name:       tc.Function.Name,
				Args:       tc.Function.Arguments,
				ArgsDigest: types.ArgsDigestHex(tc.Function.Arguments),
			})
			idx++
		}
	}
	return out
}

// evaluateResponseAction runs the ResponseAction hook (if installed) against the
// proposed calls and returns the decision plus whether governance applies. When
// hook is nil or there are no tool calls, applies is false and the caller
// releases the response unchanged.
func evaluateResponseAction(hook func(types.ResponseActionInput) types.ResponseActionDecision, in types.ResponseActionInput) (types.ResponseActionDecision, bool) {
	if hook == nil || len(in.ToolCalls) == 0 {
		return types.ResponseActionDecision{}, false
	}
	return hook(in), true
}

// buildActionEnvelope builds the action envelope for a not-all-allowed decision.
// The top-level action is "hold" if any call is held, else "block".
func buildActionEnvelope(proposed []types.ProposedToolCall, decision types.ResponseActionDecision) aionActionEnvelope {
	byIndex := make(map[int]types.ResponseActionCallDecision, len(decision.Decisions))
	for _, d := range decision.Decisions {
		byIndex[d.Index] = d
	}
	anyHold := false
	entries := make([]aionActionCallEntry, 0, len(proposed))
	for _, pc := range proposed {
		d := byIndex[pc.Index]
		action := "allow"
		switch d.Verdict {
		case types.ActionBlock:
			action = "block"
		case types.ActionHold:
			action = "hold"
			anyHold = true
		}
		entries = append(entries, aionActionCallEntry{
			ToolName:    pc.Name,
			CallID:      pc.ID,
			ArgsDigest:  pc.ArgsDigest,
			Action:      action,
			HoldID:      d.HoldID,
			PolicyClass: d.PolicyClass,
		})
	}
	top := "block"
	if anyHold {
		top = "hold"
	}
	return aionActionEnvelope{AIONAction: aionActionBody{
		Action:     top,
		ReasonCode: decision.ReasonCode,
		Calls:      entries,
	}}
}

// envelopeContentJSON marshals the envelope as the JSON string that becomes the
// governed response's assistant content.
func envelopeContentJSON(env aionActionEnvelope) json.RawMessage {
	raw, err := json.Marshal(env)
	if err != nil {
		// Fall back to a minimal valid envelope; never leak tool calls.
		return json.RawMessage(`{"aion_action":{"action":"block","calls":[]}}`)
	}
	// The content is a JSON string field, so encode the envelope as a string.
	s, err := json.Marshal(string(raw))
	if err != nil {
		return json.RawMessage(`""`)
	}
	return json.RawMessage(s)
}

// streamToolBuffer accumulates streaming tool-call fragments so a complete tool
// call can be evaluated before any is released. It is used only when a
// ResponseAction hook is installed; otherwise the stream passes through
// unchanged. Content-only and usage chunks are never buffered here (the caller
// flushes them immediately); this buffer holds only the chunks that carried a
// tool-call delta, plus the assembled calls.
type streamToolBuffer struct {
	byIndex map[int]*types.ToolCall
	order   []int
	// bufferedChunks are the raw chunks that carried tool-call deltas, kept so an
	// all-allowed decision can replay them verbatim (preserving provider ids,
	// timing splits and any interleaved content in those chunks).
	bufferedChunks []*types.ChatCompletionChunk
	overflow       bool
}

func newStreamToolBuffer() *streamToolBuffer {
	return &streamToolBuffer{byIndex: map[int]*types.ToolCall{}}
}

// chunkHasToolCall reports whether a chunk carries any tool-call delta.
func chunkHasToolCall(chunk *types.ChatCompletionChunk) bool {
	for _, c := range chunk.Choices {
		if len(c.Delta.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

// add accumulates a chunk's tool-call fragments and retains the chunk for
// possible replay. It enforces the buffered-arguments ceiling per call.
func (b *streamToolBuffer) add(chunk *types.ChatCompletionChunk) {
	b.bufferedChunks = append(b.bufferedChunks, chunk)
	for _, choice := range chunk.Choices {
		for _, tc := range choice.Delta.ToolCalls {
			idx := 0
			if tc.Index != nil {
				idx = *tc.Index
			}
			cur := b.byIndex[idx]
			if cur == nil {
				cur = &types.ToolCall{Type: "function"}
				b.byIndex[idx] = cur
				b.order = append(b.order, idx)
			}
			if tc.ID != "" {
				cur.ID = tc.ID
			}
			if tc.Function.Name != "" {
				cur.Function.Name = tc.Function.Name
			}
			cur.Function.Arguments += tc.Function.Arguments
			if len(cur.Function.Arguments) > types.MaxBufferedToolCallArgsBytes {
				b.overflow = true
			}
		}
	}
}

// hasToolCalls reports whether any tool-call fragment was buffered.
func (b *streamToolBuffer) hasToolCalls() bool { return len(b.order) > 0 }

// proposed returns the assembled complete tool calls in arrival order, or an
// error when the buffered arguments overflowed the ceiling.
func (b *streamToolBuffer) proposed() ([]types.ProposedToolCall, error) {
	if b.overflow {
		return nil, errToolArgsTooLarge
	}
	sorted := append([]int(nil), b.order...)
	sort.Ints(sorted)
	out := make([]types.ProposedToolCall, 0, len(sorted))
	for i, idx := range sorted {
		tc := b.byIndex[idx]
		out = append(out, types.ProposedToolCall{
			Index:      i,
			ID:         tc.ID,
			Name:       tc.Function.Name,
			Args:       tc.Function.Arguments,
			ArgsDigest: types.ArgsDigestHex(tc.Function.Arguments),
		})
	}
	return out, nil
}

// envelopeChunk builds a single streaming chunk carrying the action envelope as
// ordinary assistant content with finish_reason "stop", to replace withheld tool
// calls on a not-all-allowed (or overflow) decision.
func envelopeChunk(model string, proposed []types.ProposedToolCall, decision types.ResponseActionDecision) *types.ChatCompletionChunk {
	content := string(rawStringUnquote(envelopeContentJSON(buildActionEnvelope(proposed, decision))))
	stop := "stop"
	return &types.ChatCompletionChunk{
		Object: "chat.completion.chunk",
		Model:  model,
		Choices: []types.ChunkChoice{{
			Index:        0,
			Delta:        types.ChunkDelta{Role: "assistant", Content: &content},
			FinishReason: &stop,
		}},
	}
}

// rawStringUnquote turns a JSON string value (produced by envelopeContentJSON)
// back into the raw envelope JSON text for use as streaming delta content.
func rawStringUnquote(v json.RawMessage) json.RawMessage {
	var s string
	if err := json.Unmarshal(v, &s); err != nil {
		return v
	}
	return json.RawMessage(s)
}

// rewriteResponseWithEnvelope replaces the tool calls in every choice with the
// action envelope as ordinary assistant content and clears the tool calls, so no
// executable proposal reaches the client. finish_reason becomes "stop" (a
// protocol-valid ordinary completion, not "tool_calls").
func rewriteResponseWithEnvelope(resp *types.ChatCompletionResponse, proposed []types.ProposedToolCall, decision types.ResponseActionDecision) {
	if resp == nil {
		return
	}
	content := envelopeContentJSON(buildActionEnvelope(proposed, decision))
	for i := range resp.Choices {
		resp.Choices[i].Message.ToolCalls = nil
		resp.Choices[i].Message.Content = content
		resp.Choices[i].FinishReason = "stop"
	}
}
