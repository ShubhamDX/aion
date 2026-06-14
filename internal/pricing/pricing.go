package pricing

import (
	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/pkg/types"
)

// ModelPrice holds per-token pricing for a single model.
type ModelPrice struct {
	ModelID               string
	Provider              string
	InputPricePer1M       float64
	CachedInputPricePer1M float64
	CacheWritePricePer1M  float64
	OutputPricePer1M      float64
}

// CostBreakdown prices each token class at its own rate.
type CostBreakdown struct {
	UncachedInputUSD      float64
	CacheReadInputUSD     float64
	CacheCreationInputUSD float64
	OutputUSD             float64
	TotalUSD              float64
}

// Table provides model price lookups and cost estimation.
type Table struct {
	prices map[string]*ModelPrice
}

// NewTable builds a pricing table from all providers in the config.
// It iterates OpenAI, Anthropic, and OpenRouter providers and indexes
// each model by its ID.
func NewTable(providers config.ProvidersConfig) *Table {
	prices := make(map[string]*ModelPrice)

	addModels := func(providerName string, pc *config.ProviderConfig) {
		if pc == nil {
			return
		}
		for _, m := range pc.Models {
			prices[m.ID] = &ModelPrice{
				ModelID:               m.ID,
				Provider:              providerName,
				InputPricePer1M:       m.InputPricePer1M,
				CachedInputPricePer1M: m.CachedInputPricePer1M,
				CacheWritePricePer1M:  m.CacheWritePricePer1M,
				OutputPricePer1M:      m.OutputPricePer1M,
			}
		}
	}

	addModels("openai", providers.OpenAI)
	addModels("anthropic", providers.Anthropic)
	addModels("openrouter", providers.OpenRouter)
	addModels("bedrock", providers.Bedrock)
	addModels("vertex", providers.Vertex)
	addModels("gemini", providers.Gemini)
	addModels("grok", providers.Grok)

	// Local models — always $0.
	if lp := providers.Local; lp != nil && lp.Enabled {
		for _, m := range lp.Models {
			prices[m.ID] = &ModelPrice{
				ModelID:          m.ID,
				Provider:         "local",
				InputPricePer1M:  0,
				OutputPricePer1M: 0,
			}
		}
	}

	return &Table{prices: prices}
}

// Lookup returns the pricing info for a model. The second return value
// indicates whether the model was found.
func (t *Table) Lookup(modelID string) (*ModelPrice, bool) {
	p, ok := t.prices[modelID]
	return p, ok
}

// EstimateCost calculates the cost for the given token counts using
// the specified model's pricing. Returns 0 if the model is not found.
func (t *Table) EstimateCost(modelID string, inputTokens, outputTokens int) float64 {
	p, ok := t.prices[modelID]
	if !ok {
		return 0
	}
	return (float64(inputTokens)*p.InputPricePer1M + float64(outputTokens)*p.OutputPricePer1M) / 1_000_000
}

// EstimateUsageCost calculates cost from the normalized input-token partition.
func (t *Table) EstimateUsageCost(modelID string, usage types.Usage) CostBreakdown {
	p, ok := t.prices[modelID]
	if !ok {
		return CostBreakdown{}
	}
	return costForUsage(p, usage)
}

// MostExpensiveModelCost returns the cost that would be incurred if the
// most expensive model in the table were used for the given token counts.
// This is useful for calculating savings compared to a naive approach.
func (t *Table) MostExpensiveModelCost(inputTokens, outputTokens int) float64 {
	var maxCost float64
	for _, p := range t.prices {
		cost := (float64(inputTokens)*p.InputPricePer1M + float64(outputTokens)*p.OutputPricePer1M) / 1_000_000
		if cost > maxCost {
			maxCost = cost
		}
	}
	return maxCost
}

// MostExpensiveUsageCost returns the highest cache-aware cost for this usage.
func (t *Table) MostExpensiveUsageCost(usage types.Usage) float64 {
	var maxCost float64
	for _, p := range t.prices {
		cost := costForUsage(p, usage).TotalUSD
		if cost > maxCost {
			maxCost = cost
		}
	}
	return maxCost
}

// RouteCostInputs describes a multi-turn conversation shape for comparing the
// projected cost of staying on a warm model vs switching to a cheaper one.
//
//   - PrefixTokens: the reusable conversation prefix (system prompt, history)
//     that a warm provider cache would serve at the cached-read rate.
//   - PerTurnInputTokens: NEW input tokens added each turn beyond the prefix.
//   - PerTurnOutputTokens: output tokens generated each turn.
//   - RemainingTurns: how many more turns the conversation is expected to run
//     (the horizon over which a switch amortizes its one-time re-ingestion).
type RouteCostInputs struct {
	PrefixTokens        int
	PerTurnInputTokens  int
	PerTurnOutputTokens int
	RemainingTurns      int
}

// RouteComparison is the result of CompareWarmStayVsColdSwitch. Costs are the
// projected totals over RemainingTurns. SwitchSaves is true when switching is
// strictly cheaper across the horizon (i.e. the cheaper per-turn rate beats the
// one-time cold re-ingestion penalty on the warm model's cache).
type RouteComparison struct {
	WarmStayUSD   float64
	ColdSwitchUSD float64
	SwitchSaves   bool
}

// CompareWarmStayVsColdSwitch projects the cost of two routing choices over a
// multi-turn horizon, cache-aware:
//
//   - Warm stay (warmModelID already holds a warm cache for the prefix): every
//     remaining turn pays the prefix at the warm model's CACHED-READ rate, plus
//     the per-turn fresh input and output.
//   - Cold switch (coldModelID has NO warm cache): turn 1 re-ingests the whole
//     prefix COLD (at the cold model's cache-WRITE rate, which falls back to its
//     full input rate when unmodeled), then turns 2..N serve the prefix at the
//     cold model's cached-read rate. Plus per-turn fresh input and output each
//     turn.
//
// It is a pure projection used by the S3 stickiness policy to decide whether a
// cheaper tier actually wins once the cache-loss penalty is amortized. It does
// NOT make or apply a routing decision. An unknown model id yields a zero cost
// on that side (and SwitchSaves stays false unless the other side is also zero).
// Negative token fields are clamped to zero; RemainingTurns below 1 is treated
// as 1.
func (t *Table) CompareWarmStayVsColdSwitch(warmModelID, coldModelID string, in RouteCostInputs) RouteComparison {
	// Clamp token estimates to nonnegative: a negative estimate (bad caller math)
	// must never produce a negative cost that flips SwitchSaves the wrong way.
	in.PrefixTokens = max(in.PrefixTokens, 0)
	in.PerTurnInputTokens = max(in.PerTurnInputTokens, 0)
	in.PerTurnOutputTokens = max(in.PerTurnOutputTokens, 0)
	turns := in.RemainingTurns
	if turns < 1 {
		turns = 1
	}
	warm, warmOK := t.prices[warmModelID]
	cold, coldOK := t.prices[coldModelID]

	var cmp RouteComparison
	if warmOK {
		// Warm cache served at cached-read rate every turn; prefix stays warm.
		perTurn := tokenCost(in.PrefixTokens, cachedReadRate(warm)) +
			tokenCost(in.PerTurnInputTokens, warm.InputPricePer1M) +
			tokenCost(in.PerTurnOutputTokens, warm.OutputPricePer1M)
		cmp.WarmStayUSD = perTurn * float64(turns)
	}
	if coldOK {
		// Turn 1: cold re-ingestion of the prefix at the write rate.
		coldStart := tokenCost(in.PrefixTokens, cacheWriteRate(cold)) +
			tokenCost(in.PerTurnInputTokens, cold.InputPricePer1M) +
			tokenCost(in.PerTurnOutputTokens, cold.OutputPricePer1M)
		// Turns 2..N: prefix now warm on the cold model, at its cached-read rate.
		coldTail := tokenCost(in.PrefixTokens, cachedReadRate(cold)) +
			tokenCost(in.PerTurnInputTokens, cold.InputPricePer1M) +
			tokenCost(in.PerTurnOutputTokens, cold.OutputPricePer1M)
		cmp.ColdSwitchUSD = coldStart + coldTail*float64(turns-1)
	}
	cmp.SwitchSaves = warmOK && coldOK && cmp.ColdSwitchUSD < cmp.WarmStayUSD
	return cmp
}

// cachedReadRate returns the model's cached-read rate, falling back to the full
// input rate when unset (conservative: no assumed discount).
func cachedReadRate(p *ModelPrice) float64 {
	if p.CachedInputPricePer1M > 0 {
		return p.CachedInputPricePer1M
	}
	return p.InputPricePer1M
}

// cacheWriteRate returns the model's cache-write (creation) rate, falling back
// to the full input rate when unset.
func cacheWriteRate(p *ModelPrice) float64 {
	if p.CacheWritePricePer1M > 0 {
		return p.CacheWritePricePer1M
	}
	return p.InputPricePer1M
}

func tokenCost(tokens int, ratePer1M float64) float64 {
	return float64(tokens) * ratePer1M / 1_000_000
}

func costForUsage(p *ModelPrice, usage types.Usage) CostBreakdown {
	usage.NormalizeInputPartition()
	breakdown := CostBreakdown{
		UncachedInputUSD:      tokenCost(usage.UncachedInputTokens, p.InputPricePer1M),
		CacheReadInputUSD:     tokenCost(usage.CacheReadInputTokens, cachedReadRate(p)),
		CacheCreationInputUSD: tokenCost(usage.CacheCreationInputTokens, cacheWriteRate(p)),
		OutputUSD:             tokenCost(usage.CompletionTokens, p.OutputPricePer1M),
	}
	breakdown.TotalUSD = breakdown.UncachedInputUSD + breakdown.CacheReadInputUSD +
		breakdown.CacheCreationInputUSD + breakdown.OutputUSD
	return breakdown
}
