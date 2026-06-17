package proxy

import (
	"log/slog"

	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/types"
)

// dispatchPostResponse runs the PostResponse hook OFF the request goroutine, so
// the hook's work (an embedding product's evidence recording / response-shape
// validation) can never delay the customer response. Even though the handler
// writes the body before this point, a synchronous hook would still run before
// the handler returns, and net/http may hold a small buffered body until then;
// dispatching asynchronously closes that gap.
//
// in is a value (PostResponseInput has no shared mutable state; ResponseContents
// is a freshly built slice), so handing it to the goroutine is race-free. A hook
// panic is recovered here: a buggy embedding hook must not crash the proxy after
// the response was already served. nil hooks / func fields are no-ops.
func (h *Handler) dispatchPostResponse(in types.PostResponseInput) {
	if h.hooks == nil || h.hooks.PostResponse == nil {
		return
	}
	hook := h.hooks.PostResponse
	h.postResponseWG.Add(1)
	go func() {
		defer h.postResponseWG.Done()
		defer func() {
			if r := recover(); r != nil {
				slog.Error("aion: PostResponse hook panicked; response already served",
					"request_id", in.RequestID, "panic", r)
			}
		}()
		hook(in)
	}()
}

// DrainPostResponse blocks until all in-flight asynchronous PostResponse hooks
// finish. Call it during graceful shutdown, AFTER the HTTP server has stopped
// accepting requests and BEFORE the embedding product closes its evidence store,
// so a hook's signed-row write is never cut off mid-flight.
func (h *Handler) DrainPostResponse() { h.postResponseWG.Wait() }

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
