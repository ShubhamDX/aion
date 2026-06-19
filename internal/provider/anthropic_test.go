package provider

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/pkg/types"
)

// SP2b-2 golden test for a TRANSLATED provider: carrying schema settings must not
// change the Anthropic upstream payload. translateRequest builds its own struct
// field-by-field and never reads SchemaSettings, so the bytes must match.
func TestAnthropicTranslate_SchemaSettingsNeverLeak(t *testing.T) {
	p := &AnthropicProvider{}
	base := &types.ChatCompletionRequest{
		Model:    "claude-x",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	withCarrier := *base
	withCarrier.SchemaSettings = &types.SchemaSettings{
		Mode: "tool_schema", SchemaPolicyID: "sp-orders", SchemaVersion: "1",
		SchemaHash: "deadbeef", FailClosed: true, Reason: "tool_schema_supported",
	}

	bare, err := json.Marshal(p.translateRequest(base, "claude-3", false))
	if err != nil {
		t.Fatalf("marshal bare: %v", err)
	}
	carrier, err := json.Marshal(p.translateRequest(&withCarrier, "claude-3", false))
	if err != nil {
		t.Fatalf("marshal carrier: %v", err)
	}
	if string(bare) != string(carrier) {
		t.Fatalf("schema settings changed the Anthropic payload:\n bare:    %s\n carrier: %s", bare, carrier)
	}
	for _, banned := range []string{"tool_schema", "sp-orders", "deadbeef", "schema"} {
		if strings.Contains(string(carrier), banned) {
			t.Fatalf("schema setting %q leaked into Anthropic payload: %s", banned, carrier)
		}
	}
}

func TestUsageFromAnthropicIncludesCacheTokens(t *testing.T) {
	usage := usageFromAnthropic(anthropicUsage{
		InputTokens:                40,
		CacheReadInputTokens:       50,
		CacheCreationInputTokens:   9,
		CacheCreationInputTokens1h: 1,
		OutputTokens:               20,
		ServiceTier:                "ephemeral",
	})
	if usage.PromptTokens != 100 || usage.CompletionTokens != 20 || usage.TotalTokens != 120 {
		t.Fatalf("bad aggregate usage: %+v", usage)
	}
	if usage.UncachedInputTokens != 40 || usage.CacheReadInputTokens != 50 || usage.CacheCreationInputTokens != 10 {
		t.Fatalf("bad cache partition: %+v", usage)
	}
	if usage.ProviderCacheMode != "ephemeral" {
		t.Fatalf("cache mode = %q", usage.ProviderCacheMode)
	}
	if !usage.InputPartitionValid() {
		t.Fatal("usage must satisfy partition invariant")
	}
}

func TestUsageFromAnthropicOmitsPartitionWithoutCache(t *testing.T) {
	usage := usageFromAnthropic(anthropicUsage{InputTokens: 40, OutputTokens: 20})
	if usage.PromptTokens != 40 || usage.CompletionTokens != 20 || usage.TotalTokens != 60 {
		t.Fatalf("bad aggregate usage: %+v", usage)
	}
	if usage.UncachedInputTokens != 0 || usage.CacheReadInputTokens != 0 || usage.CacheCreationInputTokens != 0 {
		t.Fatalf("cache fields should stay omitted without cache: %+v", usage)
	}
}
