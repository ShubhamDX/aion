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

// SP2b-2 golden test: carrying resolved schema settings on the request must NOT
// change the upstream OpenAI-compatible payload by a single byte. The carrier is
// `json:"-"` and upstreamPayload nils it; this proves no schema setting leaks
// into the body before activation.
func TestUpstreamPayload_SchemaSettingsNeverSerialize(t *testing.T) {
	base := &types.ChatCompletionRequest{
		Model:    "aion-auto",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	withCarrier := *base
	withCarrier.SchemaSettings = &types.SchemaSettings{
		Mode:           "provider_native",
		SchemaPolicyID: "sp-orders",
		SchemaVersion:  "1",
		SchemaHash:     "deadbeef",
		FailClosed:     true,
		Reason:         "provider_native_supported",
	}

	bareBody, err := json.Marshal(upstreamPayload(base, "gpt-4o", false))
	if err != nil {
		t.Fatalf("marshal bare: %v", err)
	}
	carrierBody, err := json.Marshal(upstreamPayload(&withCarrier, "gpt-4o", false))
	if err != nil {
		t.Fatalf("marshal carrier: %v", err)
	}
	if string(bareBody) != string(carrierBody) {
		t.Fatalf("schema settings changed the upstream payload:\n bare:    %s\n carrier: %s", bareBody, carrierBody)
	}
	// Belt and suspenders: the resolved mode + identity must not appear anywhere.
	for _, banned := range []string{"provider_native", "sp-orders", "deadbeef", "SchemaSettings", "schema_settings", "FailClosed"} {
		if strings.Contains(string(carrierBody), banned) {
			t.Fatalf("schema setting %q leaked into upstream body: %s", banned, carrierBody)
		}
	}
	// upstreamPayload must not mutate the caller's request carrier.
	if withCarrier.SchemaSettings == nil || withCarrier.SchemaSettings.Mode != "provider_native" {
		t.Fatal("upstreamPayload must not mutate the caller's SchemaSettings")
	}
}

// Even marshaling the request directly (no upstreamPayload) must not emit the
// carrier: it is `json:"-"`. Guards against a future struct change dropping the
// tag.
func TestChatCompletionRequest_SchemaSettingsIsJSONIgnored(t *testing.T) {
	req := types.ChatCompletionRequest{
		Model:          "m",
		SchemaSettings: &types.SchemaSettings{Mode: "tool_schema", SchemaPolicyID: "sp-x"},
	}
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, banned := range []string{"tool_schema", "sp-x", "SchemaSettings", "schema_settings"} {
		if strings.Contains(string(body), banned) {
			t.Fatalf("carrier %q must never serialize, got: %s", banned, body)
		}
	}
}
