package types

import (
	"encoding/json"
	"testing"
)

func TestUsageNormalizeOpenAICachedTokens(t *testing.T) {
	var usage Usage
	raw := []byte(`{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":60}}`)
	if err := json.Unmarshal(raw, &usage); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	usage.NormalizeInputPartition()
	if usage.UncachedInputTokens != 40 || usage.CacheReadInputTokens != 60 || usage.CacheCreationInputTokens != 0 {
		t.Fatalf("bad partition: %+v", usage)
	}
	if !usage.InputPartitionValid() {
		t.Fatal("normalized usage must satisfy partition invariant")
	}
}

func TestUsageInvalidPartition(t *testing.T) {
	usage := Usage{
		PromptTokens:             100,
		CompletionTokens:         20,
		TotalTokens:              120,
		UncachedInputTokens:      40,
		CacheReadInputTokens:     60,
		CacheCreationInputTokens: 1,
	}
	if usage.InputPartitionValid() {
		t.Fatal("mismatched partition must fail")
	}
}

func TestUsageMergeStreamingChunks(t *testing.T) {
	var total Usage
	total.MergeFrom(Usage{PromptTokens: 100, UncachedInputTokens: 40, CacheReadInputTokens: 60})
	total.MergeFrom(Usage{CompletionTokens: 20, TotalTokens: 20})
	if total.PromptTokens != 100 || total.CompletionTokens != 20 || total.TotalTokens != 120 {
		t.Fatalf("bad merged totals: %+v", total)
	}
	if !total.InputPartitionValid() {
		t.Fatal("merged usage must satisfy partition invariant")
	}
}
