package pricing

import (
	"math"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/pkg/types"
)

func TestEstimateUsageCostUsesCacheRates(t *testing.T) {
	table := NewTable(config.ProvidersConfig{
		OpenAI: &config.ProviderConfig{Models: []config.ModelConfig{{
			ID: "m1", InputPricePer1M: 10, CachedInputPricePer1M: 1,
			CacheWritePricePer1M: 12, OutputPricePer1M: 20,
		}}},
	})
	cost := table.EstimateUsageCost("m1", types.Usage{
		PromptTokens:             100,
		CompletionTokens:         10,
		UncachedInputTokens:      40,
		CacheReadInputTokens:     50,
		CacheCreationInputTokens: 10,
	})
	if !closeFloat(cost.UncachedInputUSD, 0.0004) || !closeFloat(cost.CacheReadInputUSD, 0.00005) ||
		!closeFloat(cost.CacheCreationInputUSD, 0.00012) || !closeFloat(cost.OutputUSD, 0.0002) {
		t.Fatalf("bad breakdown: %+v", cost)
	}
	if !closeFloat(cost.TotalUSD, 0.00077) {
		t.Fatalf("total = %.8f", cost.TotalUSD)
	}
}

func TestEstimateUsageCostFallsBackToInputRate(t *testing.T) {
	table := NewTable(config.ProvidersConfig{
		OpenAI: &config.ProviderConfig{Models: []config.ModelConfig{{
			ID: "m1", InputPricePer1M: 10, OutputPricePer1M: 20,
		}}},
	})
	cost := table.EstimateUsageCost("m1", types.Usage{
		PromptTokens:         100,
		CompletionTokens:     10,
		CacheReadInputTokens: 50,
	})
	if !closeFloat(cost.TotalUSD, table.EstimateCost("m1", 100, 10)) {
		t.Fatalf("fallback total = %.8f, want %.8f", cost.TotalUSD, table.EstimateCost("m1", 100, 10))
	}
}

func closeFloat(a, b float64) bool {
	return math.Abs(a-b) < 0.000000000001
}
