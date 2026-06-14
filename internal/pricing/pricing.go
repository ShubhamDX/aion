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

func costForUsage(p *ModelPrice, usage types.Usage) CostBreakdown {
	usage.NormalizeInputPartition()
	cachedRate := p.CachedInputPricePer1M
	if cachedRate == 0 {
		cachedRate = p.InputPricePer1M
	}
	writeRate := p.CacheWritePricePer1M
	if writeRate == 0 {
		writeRate = p.InputPricePer1M
	}
	breakdown := CostBreakdown{
		UncachedInputUSD:      float64(usage.UncachedInputTokens) * p.InputPricePer1M / 1_000_000,
		CacheReadInputUSD:     float64(usage.CacheReadInputTokens) * cachedRate / 1_000_000,
		CacheCreationInputUSD: float64(usage.CacheCreationInputTokens) * writeRate / 1_000_000,
		OutputUSD:             float64(usage.CompletionTokens) * p.OutputPricePer1M / 1_000_000,
	}
	breakdown.TotalUSD = breakdown.UncachedInputUSD + breakdown.CacheReadInputUSD +
		breakdown.CacheCreationInputUSD + breakdown.OutputUSD
	return breakdown
}
