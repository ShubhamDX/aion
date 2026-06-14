package types

import (
	"encoding/json"
	"testing"
)

// msg is defined in gatewayhooks_test.go (same package).

func req(msgs ...Message) *ChatCompletionRequest {
	return &ChatCompletionRequest{Model: "m", Messages: msgs}
}

func TestSessionMaterial_HeaderWinsOverBody(t *testing.T) {
	r := req(msg("system", "sys"), msg("user", "hi"))
	r.AIONPreferences = &AIONPreferences{SessionID: "body-id"}
	m := SessionMaterialFromRequest(r, "header-id")
	if m.Source != SessionSourceHeader {
		t.Fatalf("source = %q, want header", m.Source)
	}
	if m.SessionMaterialSHA256 == "" {
		t.Fatal("header id must yield a digest")
	}
	// The header value, not the body value, must drive the digest.
	wantHeader := SessionMaterialFromRequest(req(msg("user", "x")), "header-id")
	if m.SessionMaterialSHA256 != wantHeader.SessionMaterialSHA256 {
		t.Fatal("digest must derive from the header id, independent of body/messages")
	}
}

func reqWithBody(id string) *ChatCompletionRequest {
	r := req(msg("user", "hi"))
	r.AIONPreferences = &AIONPreferences{SessionID: id}
	return r
}

func TestSessionMaterial_BodyWhenNoHeader(t *testing.T) {
	m := SessionMaterialFromRequest(reqWithBody("body-id"), "")
	if m.Source != SessionSourceBody {
		t.Fatalf("source = %q, want body", m.Source)
	}
	if m.SessionMaterialSHA256 == "" {
		t.Fatal("body id must yield a digest")
	}
}

func TestSessionMaterial_DerivedFromStableRoot(t *testing.T) {
	r := req(msg("system", "you are helpful"), msg("user", "first question"))
	m := SessionMaterialFromRequest(r, "")
	if m.Source != SessionSourceDerived {
		t.Fatalf("source = %q, want derived", m.Source)
	}
	if m.SessionMaterialSHA256 == "" {
		t.Fatal("derived root must yield a digest")
	}
	// Same stable root on a later turn (more messages appended) must derive the
	// SAME session digest: identity is stable across the conversation.
	r2 := req(msg("system", "you are helpful"), msg("user", "first question"),
		msg("assistant", "answer"), msg("user", "second question"))
	m2 := SessionMaterialFromRequest(r2, "")
	if m2.SessionMaterialSHA256 != m.SessionMaterialSHA256 {
		t.Fatal("derived session digest must be stable across turns")
	}
}

func TestSessionMaterial_NoStableRootIsNoKey(t *testing.T) {
	// No system msg, no user msg (only an assistant msg): nothing stable to anchor.
	m := SessionMaterialFromRequest(req(msg("assistant", "orphan")), "")
	if m.Source != SessionSourceNone {
		t.Fatalf("source = %q, want no_key", m.Source)
	}
	if m.SessionMaterialSHA256 != "" {
		t.Fatal("no_key must carry no digest")
	}
}

func TestSessionMaterial_PrefixOnlyWithPriorTurn(t *testing.T) {
	// Single message: no prior prefix to reuse.
	if got := SessionMaterialFromRequest(req(msg("user", "hi")), "x").CachePrefixMaterialSHA256; got != "" {
		t.Fatalf("single message must have no cache prefix, got %q", got)
	}
	// Two+ messages: prefix is everything but the last.
	m := SessionMaterialFromRequest(req(msg("system", "s"), msg("user", "u1"), msg("user", "u2")), "x")
	if m.CachePrefixMaterialSHA256 == "" {
		t.Fatal("multi-message request must have a cache prefix digest")
	}
}

func TestNextCachePrefixMaterial_AvailableVsStreaming(t *testing.T) {
	r := req(msg("system", "s"), msg("user", "u1"))
	resp := &ChatCompletionResponse{Choices: []Choice{{
		Message: Message{Role: "assistant", Content: mustJSON("answer")},
	}}}
	next := NextCachePrefixMaterial(r, resp)
	if next == "" {
		t.Fatal("next prefix must be computed when the response is available")
	}
	// The next-turn prefix (req messages + assistant) must equal THIS turn's full
	// message digest on the following request, i.e. it chains.
	r2 := req(msg("system", "s"), msg("user", "u1"), msg("assistant", "answer"))
	if next != messagesDigest(r2.Messages) {
		t.Fatal("next prefix digest must chain to the following turn's full prefix")
	}
	// Streaming / no response -> "".
	if NextCachePrefixMaterial(r, nil) != "" {
		t.Fatal("nil response must yield empty next prefix (safe-degrade)")
	}
	if NextCachePrefixMaterial(r, &ChatCompletionResponse{}) != "" {
		t.Fatal("response with no choices must yield empty next prefix")
	}
}

func mustJSON(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

func TestCachePrefix_CutsAtLastAssistant_AgenticTurn(t *testing.T) {
	// Turn N: system, user, assistant(tool_calls). Stored next-prefix = those 3.
	rN := req(msg("system", "s"), msg("user", "u1"))
	respN := &ChatCompletionResponse{Choices: []Choice{{
		Message: Message{Role: "assistant", Content: mustJSON("calling tools")},
	}}}
	next := NextCachePrefixMaterial(rN, respN)
	if next == "" {
		t.Fatal("next prefix must be set")
	}

	// Turn N+1: client appends TWO tool-result messages after the assistant.
	// This-turn prefix must cut at the assistant (the 3-message head), NOT at
	// len-1 (which would wrongly include the first tool result), so it equals the
	// stored next-prefix and the warm cache is found.
	rN1 := req(msg("system", "s"), msg("user", "u1"), msg("assistant", "calling tools"),
		msg("tool", "result-a"), msg("tool", "result-b"))
	m := SessionMaterialFromRequest(rN1, "sess")
	if m.CachePrefixMaterialSHA256 != next {
		t.Fatal("agentic this-turn prefix must chain to the stored next-prefix (cut at last assistant)")
	}
}

func TestCachePrefix_FirstTurnFallback(t *testing.T) {
	// No assistant yet, system + user: fall back to all-but-last so a multi-message
	// first turn still has a prefix.
	m := SessionMaterialFromRequest(req(msg("system", "s"), msg("user", "u1")), "x")
	if m.CachePrefixMaterialSHA256 == "" {
		t.Fatal("multi-message first turn must still have a prefix")
	}
	// Bare single user message: no reusable prefix.
	if got := SessionMaterialFromRequest(req(msg("user", "hi")), "x").CachePrefixMaterialSHA256; got != "" {
		t.Fatalf("single message must have no prefix, got %q", got)
	}
}

func TestScrubSessionID(t *testing.T) {
	r := reqWithBody("secret-thread-42")
	r.AIONPreferences.PreferredTier = ptrInt(2)
	ScrubSessionID(r)
	if r.AIONPreferences.SessionID != "" {
		t.Fatal("session id must be scrubbed")
	}
	if r.AIONPreferences.PreferredTier == nil || *r.AIONPreferences.PreferredTier != 2 {
		t.Fatal("other routing hints must survive scrub")
	}
	// No prefs -> no panic.
	ScrubSessionID(req(msg("user", "hi")))
}

func ptrInt(i int) *int { return &i }
