package pricing

import (
	"github.com/ShubhamDX/aion/internal/config"
)

// ModelPrice holds per-token pricing for a single model.
type ModelPrice struct {
	ModelID          string
	Provider         string
	InputPricePer1M  float64
	OutputPricePer1M float64
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
				ModelID:          m.ID,
				Provider:         providerName,
				InputPricePer1M:  m.InputPricePer1M,
				OutputPricePer1M: m.OutputPricePer1M,
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
