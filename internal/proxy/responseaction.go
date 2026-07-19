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
// choices of a non-streaming response, in stable index order. Index is the
// response-wide 0..N-1 position; ChoiceIndex/CallIndex carry the provider-scoped
// coordinates so an n>1 response keeps two choices' tool index 0 distinct.
func proposedToolCallsFromResponse(resp *types.ChatCompletionResponse) []types.ProposedToolCall {
	if resp == nil {
		return nil
	}
	var out []types.ProposedToolCall
	idx := 0
	for ci, choice := range resp.Choices {
		for cj, tc := range choice.Message.ToolCalls {
			out = append(out, types.ProposedToolCall{
				Index:       idx,
				ChoiceIndex: ci,
				CallIndex:   cj,
				ID:          tc.ID,
				Name:        tc.Function.Name,
				Args:        tc.Function.Arguments,
				ArgsDigest:  types.ArgsDigestHex(tc.Function.Arguments),
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

// buildActionEnvelope builds the action envelope for a not-all-allowed (or
// malformed) decision. The top-level action uses deterministic severity ordering
// (block > hold > allow); a malformed decision fails closed to "block". Each
// per-call entry reflects that call's own verdict; a call with no valid decision
// is rendered "block" so the client never sees an unclassified call as allowed.
func buildActionEnvelope(proposed []types.ProposedToolCall, decision types.ResponseActionDecision) aionActionEnvelope {
	valid := decision.Validate(len(proposed))
	byIndex := make(map[int]types.ResponseActionCallDecision, len(decision.Decisions))
	if valid {
		for _, d := range decision.Decisions {
			byIndex[d.Index] = d
		}
	}
	entries := make([]aionActionCallEntry, 0, len(proposed))
	for _, pc := range proposed {
		d, ok := byIndex[pc.Index]
		// Fail closed: any call without a valid decision is blocked, never allowed.
		action := "block"
		if ok {
			switch d.Verdict {
			case types.ActionAllow:
				action = "allow"
			case types.ActionHold:
				action = "hold"
			case types.ActionBlock:
				action = "block"
			}
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
	return aionActionEnvelope{AIONAction: aionActionBody{
		Action:     decision.TopSeverityAction(len(proposed)),
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

// streamCallKey is the provider-scoped identity of one streaming tool call:
// (choice.index, tool_call.index). Tool-call indices are scoped to a response
// choice, so with n>1 two choices can both use tool index 0; keying by this pair
// keeps them distinct instead of merging their names and arguments.
type streamCallKey struct {
	choice int
	call   int
}

// streamCall is one assembled call plus the arrival order that fixes its stable
// response-wide Index.
type streamCall struct {
	key streamCallKey
	tc  types.ToolCall
	seq int
}

// streamToolBuffer accumulates streaming tool-call fragments so a complete tool
// call can be evaluated before any is released. It is used only when a
// ResponseAction hook is installed; otherwise the stream passes through
// unchanged. Content-only and usage chunks are never buffered here (the caller
// flushes them immediately); this buffer holds only the chunks that carried a
// tool-call delta, plus the assembled calls.
//
// Every retained dimension is bounded (per-call args, call count, aggregate args,
// raw chunk bytes). The instant any ceiling is exceeded, overflow is set and the
// buffer STOPS retaining new chunks and new argument bytes — the caller keeps
// draining the upstream stream only to close it safely, and the whole response
// fails closed to block.
type streamToolBuffer struct {
	byKey map[streamCallKey]*streamCall
	order []streamCallKey
	// bufferedChunks are the raw chunks that carried tool-call deltas, kept so an
	// all-allowed decision can replay them verbatim (preserving provider ids,
	// timing splits and any interleaved content in those chunks).
	bufferedChunks []*types.ChatCompletionChunk
	seq            int
	argsTotal      int
	chunkBytes     int
	overflow       bool
	// failReason records WHY the buffer failed closed (memory ceiling vs a
	// malformed tool-call identity), surfaced as the envelope reason_code. The
	// first failure wins.
	failReason string
}

func newStreamToolBuffer() *streamToolBuffer {
	return &streamToolBuffer{byKey: map[streamCallKey]*streamCall{}}
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

// failClosed trips the fail-closed flag with a reason and drops any
// already-retained chunks so retained memory stops growing (and shrinks) once a
// limit is hit or an identity is untrustworthy. The first reason wins.
func (b *streamToolBuffer) failClosed(reason string) {
	if b.failReason == "" {
		b.failReason = reason
	}
	b.overflow = true
	b.bufferedChunks = nil
}

// markOverflow is the memory-ceiling fail-closed.
func (b *streamToolBuffer) markOverflow() { b.failClosed("tool_args_too_large") }

// add accumulates a chunk's tool-call fragments and retains the chunk for
// possible replay, enforcing every memory ceiling and the tool-call identity
// contract. After any failure it retains nothing further.
//
// Identity is fail-closed: a streaming tool fragment MUST carry a non-nil,
// non-negative tool_call.index (the provider stream is outside the trust
// boundary, so a missing index is not silently coerced to 0 — that would collapse
// two distinct same-choice calls into one governance decision), and once a
// (choice, index) key is established its non-empty ID and function name may not
// change to a conflicting non-empty value. Either violation fails the whole
// response closed to block before the hook runs.
func (b *streamToolBuffer) add(chunk *types.ChatCompletionChunk) {
	if b.overflow {
		return // stop retaining new data immediately after any failure
	}
	// Bound retained raw chunk bytes (includes any interleaved content).
	if data, err := json.Marshal(chunk); err == nil {
		b.chunkBytes += len(data)
		if b.chunkBytes > types.MaxBufferedStreamChunkBytes {
			b.markOverflow()
			return
		}
	}
	b.bufferedChunks = append(b.bufferedChunks, chunk)
	for _, choice := range chunk.Choices {
		for _, tc := range choice.Delta.ToolCalls {
			// Fail closed on an absent or negative tool-call index: without a valid
			// index we cannot prove this fragment is a distinct call rather than a
			// continuation of another, so we refuse rather than guess.
			if tc.Index == nil || *tc.Index < 0 {
				b.failClosed("tool_call_identity_invalid")
				return
			}
			key := streamCallKey{choice: choice.Index, call: *tc.Index}
			cur := b.byKey[key]
			if cur == nil {
				if len(b.byKey) >= types.MaxBufferedToolCallCount {
					b.markOverflow()
					return
				}
				cur = &streamCall{key: key, tc: types.ToolCall{Type: "function"}, seq: b.seq}
				b.seq++
				b.byKey[key] = cur
				b.order = append(b.order, key)
			}
			// A non-empty ID or name that conflicts with the established value for
			// this key means two different calls collided on one identity: fail
			// closed rather than let the later value overwrite the earlier one.
			if tc.ID != "" {
				if cur.tc.ID != "" && cur.tc.ID != tc.ID {
					b.failClosed("tool_call_identity_conflict")
					return
				}
				cur.tc.ID = tc.ID
			}
			if tc.Function.Name != "" {
				if cur.tc.Function.Name != "" && cur.tc.Function.Name != tc.Function.Name {
					b.failClosed("tool_call_identity_conflict")
					return
				}
				cur.tc.Function.Name = tc.Function.Name
			}
			frag := tc.Function.Arguments
			cur.tc.Function.Arguments += frag
			b.argsTotal += len(frag)
			if len(cur.tc.Function.Arguments) > types.MaxBufferedToolCallArgsBytes ||
				b.argsTotal > types.MaxBufferedToolCallArgsTotalBytes {
				b.markOverflow()
				return
			}
		}
	}
}

// hasToolCalls reports whether any tool-call fragment was buffered (or overflowed
// while buffering tool calls). Overflow still counts so the caller fails closed.
func (b *streamToolBuffer) hasToolCalls() bool { return len(b.order) > 0 || b.overflow }

// failedReasonCode returns the envelope reason_code for a failed-closed buffer,
// defaulting to the memory-ceiling reason.
func (b *streamToolBuffer) failedReasonCode() string {
	if b.failReason != "" {
		return b.failReason
	}
	return "tool_args_too_large"
}

// proposed returns the assembled complete tool calls, or an error when the buffer
// failed closed (a memory ceiling or an invalid/conflicting tool-call identity).
// Calls are ordered by (choice.index, tool_call.index) and numbered 0..N-1 in
// that stable order.
func (b *streamToolBuffer) proposed() ([]types.ProposedToolCall, error) {
	if b.overflow {
		return nil, errToolArgsTooLarge
	}
	keys := append([]streamCallKey(nil), b.order...)
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].choice != keys[j].choice {
			return keys[i].choice < keys[j].choice
		}
		return keys[i].call < keys[j].call
	})
	out := make([]types.ProposedToolCall, 0, len(keys))
	for i, key := range keys {
		c := b.byKey[key]
		out = append(out, types.ProposedToolCall{
			Index:       i,
			ChoiceIndex: key.choice,
			CallIndex:   key.call,
			ID:          c.tc.ID,
			Name:        c.tc.Function.Name,
			Args:        c.tc.Function.Arguments,
			ArgsDigest:  types.ArgsDigestHex(c.tc.Function.Arguments),
		})
	}
	return out, nil
}

// failClosedEnvelope is the forced-block envelope emitted when the proxy cannot
// assemble or trust the proposed calls at all (overflow, upstream read error).
// Its top-level action is always "block" regardless of any decision, so a client
// never sees such a response as allowed.
func failClosedEnvelope(reasonCode string) aionActionEnvelope {
	return aionActionEnvelope{AIONAction: aionActionBody{
		Action:     "block",
		ReasonCode: reasonCode,
		Calls:      []aionActionCallEntry{},
	}}
}

// envelopeChunk builds a single streaming chunk carrying the action envelope as
// ordinary assistant content with finish_reason "stop", to replace withheld tool
// calls on a not-all-allowed decision.
func envelopeChunk(model string, proposed []types.ProposedToolCall, decision types.ResponseActionDecision) *types.ChatCompletionChunk {
	return envelopeChunkFrom(model, buildActionEnvelope(proposed, decision))
}

// failClosedEnvelopeChunk builds the forced-block streaming chunk for the cases
// where no trustworthy call set exists.
func failClosedEnvelopeChunk(model, reasonCode string) *types.ChatCompletionChunk {
	return envelopeChunkFrom(model, failClosedEnvelope(reasonCode))
}

func envelopeChunkFrom(model string, env aionActionEnvelope) *types.ChatCompletionChunk {
	content := string(rawStringUnquote(envelopeContentJSON(env)))
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
