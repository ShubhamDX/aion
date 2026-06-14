package proxy

import (
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/types"
)

func (h *Handler) costAndSavings(modelID string, usage types.Usage) (pricing.CostBreakdown, float64) {
	usage.NormalizeInputPartition()
	cost := h.pricing.EstimateUsageCost(modelID, usage)
	maxCost := h.pricing.MostExpensiveUsageCost(usage)
	savings := maxCost - cost.TotalUSD
	if savings < 0 {
		savings = 0
	}
	return cost, savings
}

func postResponseInputWithUsage(base types.PostResponseInput, usage types.Usage, cost pricing.CostBreakdown) types.PostResponseInput {
	usage.NormalizeInputPartition()
	base.InputTokens = usage.PromptTokens
	base.OutputTokens = usage.CompletionTokens
	base.UncachedInputTokens = usage.UncachedInputTokens
	base.CacheReadInputTokens = usage.CacheReadInputTokens
	base.CacheCreationInputTokens = usage.CacheCreationInputTokens
	base.ProviderCacheMode = usage.ProviderCacheMode
	base.UncachedInputCostUSD = cost.UncachedInputUSD
	base.CacheReadInputCostUSD = cost.CacheReadInputUSD
	base.CacheCreationInputCostUSD = cost.CacheCreationInputUSD
	base.OutputCostUSD = cost.OutputUSD
	return base
}
