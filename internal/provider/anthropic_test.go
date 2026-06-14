package provider

import "testing"

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
