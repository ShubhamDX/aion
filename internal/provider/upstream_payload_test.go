package provider

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/pkg/types"
)

// The upstream body sent to OpenAI-compatible providers MUST NOT carry AION
// control fields: aion_preferences is internal routing/session state, and
// session_id is caller-private. Shipping either leaks data and sends a
// non-standard field upstream.
func TestUpstreamPayload_StripsAIONPreferences(t *testing.T) {
	tier := 2
	req := &types.ChatCompletionRequest{
		Model:    "aion-auto",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		AIONPreferences: &types.AIONPreferences{
			SessionID:     "secret-thread-42",
			PreferredTier: &tier,
		},
	}
	payload := upstreamPayload(req, "gpt-4o", false)

	if payload.AIONPreferences != nil {
		t.Fatal("aion_preferences must be nil on the upstream payload")
	}
	if payload.Model != "gpt-4o" || payload.Stream != false {
		t.Fatalf("model/stream not set: %+v", payload)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(body)
	if strings.Contains(s, "aion_preferences") {
		t.Fatalf("aion_preferences leaked into upstream body: %s", s)
	}
	if strings.Contains(s, "secret-thread-42") || strings.Contains(s, "session_id") {
		t.Fatalf("session id leaked into upstream body: %s", s)
	}

	// The caller's request must be untouched (helper copies, not mutates).
	if req.AIONPreferences == nil || req.AIONPreferences.SessionID != "secret-thread-42" {
		t.Fatal("upstreamPayload must not mutate the caller's request")
	}
}

func TestUpstreamPayload_StreamFlag(t *testing.T) {
	req := &types.ChatCompletionRequest{Model: "x"}
	if !upstreamPayload(req, "m", true).Stream {
		t.Fatal("stream flag must propagate")
	}
}
