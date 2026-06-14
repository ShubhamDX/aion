package types

import (
	"crypto/sha256"
	"encoding/hex"
)

// SessionSource names where a request's session identity came from. It is a
// non-sensitive reason field safe to log; the actual identity is carried only as
// a one-way digest (see SessionMaterial).
type SessionSource string

const (
	// SessionSourceHeader: the X-AION-Session-Id header supplied the identity.
	SessionSourceHeader SessionSource = "header"
	// SessionSourceBody: aion_preferences.session_id supplied the identity.
	SessionSourceBody SessionSource = "body"
	// SessionSourceDerived: no explicit id; identity derived from a stable
	// conversation root (system prompt + first user message).
	SessionSourceDerived SessionSource = "derived"
	// SessionSourceNone: no stable identity could be derived; the caller must
	// safe-degrade (treat as a new conversation, route freely).
	SessionSourceNone SessionSource = "no_key"
)

// SessionMaterial is the privacy-preserving session/cache identity OSS hands to
// the embedding product. It carries ONLY sha256 digests of the canonical
// material plus a non-sensitive source reason. It never carries the raw session
// id or any prompt/response text. The embedding product re-keys these digests
// into tenant-scoped, purpose-separated HMACs; raw material never crosses the
// boundary and is never logged here.
//
// Fields are "" when unavailable:
//   - SessionMaterialSHA256: digest of the session identity (header/body id, or
//     the stable conversation root). "" only when Source is SessionSourceNone.
//   - CachePrefixMaterialSHA256: digest of THIS turn's reusable prefix (the
//     messages minus the last user turn), i.e. what a warm provider cache would
//     serve. "" when there is no prior turn to form a prefix.
//   - NextCachePrefixMaterialSHA256: digest of the NEXT turn's prefix (this
//     request's messages plus the assistant response). Computed only when the
//     full response body is available; "" on streaming (do NOT buffer a stream
//     just to compute it) or any other unavailability.
type SessionMaterial struct {
	Source                        SessionSource
	SessionMaterialSHA256         string
	CachePrefixMaterialSHA256     string
	NextCachePrefixMaterialSHA256 string
}

// SessionMaterialFromRequest extracts the pre-dispatch session material: the
// session-identity digest and THIS turn's cache-prefix digest. headerID is the
// X-AION-Session-Id header value ("" if absent). It never returns or logs raw
// material.
//
// Identity precedence: header id, then body aion_preferences.session_id, then a
// derived stable root (system prompt + first user message). When none yields a
// stable key, Source is SessionSourceNone and digests are "".
func SessionMaterialFromRequest(req *ChatCompletionRequest, headerID string) SessionMaterial {
	var m SessionMaterial
	switch {
	case headerID != "":
		m.Source = SessionSourceHeader
		m.SessionMaterialSHA256 = sha256Hex("id\x00" + headerID)
	case req != nil && req.AIONPreferences != nil && req.AIONPreferences.SessionID != "":
		m.Source = SessionSourceBody
		m.SessionMaterialSHA256 = sha256Hex("id\x00" + req.AIONPreferences.SessionID)
	default:
		if root, ok := stableConversationRoot(req); ok {
			m.Source = SessionSourceDerived
			m.SessionMaterialSHA256 = root
		} else {
			m.Source = SessionSourceNone
		}
	}
	// This turn's reusable prefix is the messages minus the last (current) user
	// turn. With 0 or 1 message there is no prior prefix to reuse.
	if req != nil && len(req.Messages) > 1 {
		m.CachePrefixMaterialSHA256 = messagesDigest(req.Messages[:len(req.Messages)-1])
	}
	return m
}

// NextCachePrefixMaterial returns the digest of the NEXT turn's reusable prefix:
// this request's messages plus the assistant response appended. resp may be nil
// (or have no choices) when the response is unavailable (streaming, error), in
// which case "" is returned and the caller safe-degrades. It never buffers or
// reassembles a stream.
func NextCachePrefixMaterial(req *ChatCompletionRequest, resp *ChatCompletionResponse) string {
	if req == nil || resp == nil || len(resp.Choices) == 0 {
		return ""
	}
	next := make([]governedMessage, 0, len(req.Messages)+1)
	for _, m := range req.Messages {
		next = append(next, governedMessage{
			Role: m.Role, Content: m.Content, Name: m.Name,
			ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID,
		})
	}
	choice := resp.Choices[0]
	next = append(next, governedMessage{
		Role: "assistant", Content: choice.Message.Content, ToolCalls: choice.Message.ToolCalls,
	})
	return digestJSON(next)
}

// stableConversationRoot digests the parts of a conversation that do not change
// across turns: the system message(s) and the first user message. Returns false
// when neither exists (no stable anchor, so no derived key).
func stableConversationRoot(req *ChatCompletionRequest) (string, bool) {
	if req == nil {
		return "", false
	}
	var root []governedMessage
	var sawFirstUser bool
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			root = append(root, governedMessage{Role: m.Role, Content: m.Content, Name: m.Name})
		case "user":
			if !sawFirstUser {
				root = append(root, governedMessage{Role: m.Role, Content: m.Content, Name: m.Name})
				sawFirstUser = true
			}
		}
	}
	if len(root) == 0 {
		return "", false
	}
	return digestJSON(root), true
}

// messagesDigest digests a slice of messages over their governed projection.
func messagesDigest(msgs []Message) string {
	g := make([]governedMessage, 0, len(msgs))
	for _, m := range msgs {
		g = append(g, governedMessage{
			Role: m.Role, Content: m.Content, Name: m.Name,
			ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID,
		})
	}
	return digestJSON(g)
}

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}
