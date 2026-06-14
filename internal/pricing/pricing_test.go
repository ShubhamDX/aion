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

// twoModelTable: a pricey warm model (tier 2) and a cheaper cold model (tier 1),
// both with explicit cache rates, for the warm-stay vs cold-switch projection.
func twoModelTable() *Table {
	return NewTable(config.ProvidersConfig{
		Anthropic: &config.ProviderConfig{Models: []config.ModelConfig{
			{ID: "sonnet", Tier: 2, InputPricePer1M: 3, CachedInputPricePer1M: 0.30, CacheWritePricePer1M: 3.75, OutputPricePer1M: 15},
			{ID: "haiku", Tier: 1, InputPricePer1M: 0.80, CachedInputPricePer1M: 0.08, CacheWritePricePer1M: 1.00, OutputPricePer1M: 4},
		}},
	})
}

func TestCompareWarmStayVsColdSwitch_LongHorizonSwitchWins(t *testing.T) {
	tbl := twoModelTable()
	// Big prefix, many remaining turns: the cheaper haiku amortizes its one-time
	// cold re-ingestion and wins overall.
	cmp := tbl.CompareWarmStayVsColdSwitch("sonnet", "haiku", RouteCostInputs{
		PrefixTokens: 20000, PerTurnInputTokens: 200, PerTurnOutputTokens: 300, RemainingTurns: 20,
	})
	if !cmp.SwitchSaves {
		t.Fatalf("long horizon should favor switching: warm=%.6f cold=%.6f", cmp.WarmStayUSD, cmp.ColdSwitchUSD)
	}
	if cmp.ColdSwitchUSD >= cmp.WarmStayUSD {
		t.Fatalf("cold switch must be cheaper: warm=%.6f cold=%.6f", cmp.WarmStayUSD, cmp.ColdSwitchUSD)
	}
}

func TestCompareWarmStayVsColdSwitch_ShortHorizonStayWins(t *testing.T) {
	tbl := twoModelTable()
	// Big prefix, only one remaining turn: the cold re-ingestion penalty on the
	// switch turn outweighs the cheaper per-turn rate, so staying warm wins.
	cmp := tbl.CompareWarmStayVsColdSwitch("sonnet", "haiku", RouteCostInputs{
		PrefixTokens: 20000, PerTurnInputTokens: 200, PerTurnOutputTokens: 300, RemainingTurns: 1,
	})
	if cmp.SwitchSaves {
		t.Fatalf("single-turn horizon should favor staying: warm=%.6f cold=%.6f", cmp.WarmStayUSD, cmp.ColdSwitchUSD)
	}
}

func TestCompareWarmStayVsColdSwitch_UnknownModelNoSwitch(t *testing.T) {
	tbl := twoModelTable()
	cmp := tbl.CompareWarmStayVsColdSwitch("sonnet", "nope", RouteCostInputs{
		PrefixTokens: 20000, RemainingTurns: 20,
	})
	if cmp.SwitchSaves {
		t.Fatal("unknown cold model must never report a saving")
	}
	if cmp.ColdSwitchUSD != 0 {
		t.Fatalf("unknown cold model cost should be 0, got %.6f", cmp.ColdSwitchUSD)
	}
}

func TestCompareWarmStayVsColdSwitch_NegativeTokensClamped(t *testing.T) {
	tbl := twoModelTable()
	// Negative estimates must clamp to zero, never produce a negative cost. With
	// all token fields negative, both projected costs are 0 and nothing "saves".
	cmp := tbl.CompareWarmStayVsColdSwitch("sonnet", "haiku", RouteCostInputs{
		PrefixTokens: -100, PerTurnInputTokens: -5, PerTurnOutputTokens: -5, RemainingTurns: 5,
	})
	if cmp.WarmStayUSD != 0 || cmp.ColdSwitchUSD != 0 {
		t.Fatalf("negative tokens must clamp to zero cost: warm=%.6f cold=%.6f", cmp.WarmStayUSD, cmp.ColdSwitchUSD)
	}
	if cmp.SwitchSaves {
		t.Fatal("no saving when both costs are zero")
	}
}

func TestCompareWarmStayVsColdSwitch_UnsetCacheRatesPriceAtFullInput(t *testing.T) {
	// A cold model with NO cache rates prices its warm tail (turns 2..N) at full
	// input rate, not a discount. Confirm by comparing against a model that is
	// identical except for the cache rates: the no-cache one must cost more.
	tbl := NewTable(config.ProvidersConfig{
		Anthropic: &config.ProviderConfig{Models: []config.ModelConfig{
			{ID: "warm", Tier: 2, InputPricePer1M: 3, CachedInputPricePer1M: 0.30, OutputPricePer1M: 15},
			{ID: "cold-cached", Tier: 1, InputPricePer1M: 0.80, CachedInputPricePer1M: 0.08, OutputPricePer1M: 4},
			{ID: "cold-nocache", Tier: 1, InputPricePer1M: 0.80, OutputPricePer1M: 4},
		}},
	})
	in := RouteCostInputs{PrefixTokens: 20000, PerTurnInputTokens: 200, PerTurnOutputTokens: 300, RemainingTurns: 10}
	cached := tbl.CompareWarmStayVsColdSwitch("warm", "cold-cached", in)
	nocache := tbl.CompareWarmStayVsColdSwitch("warm", "cold-nocache", in)
	if !(nocache.ColdSwitchUSD > cached.ColdSwitchUSD) {
		t.Fatalf("unset cache rates must price higher: nocache=%.6f cached=%.6f", nocache.ColdSwitchUSD, cached.ColdSwitchUSD)
	}
}
