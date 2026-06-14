package proxy

import (
	"encoding/json"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/pkg/types"
)

// cachedModelHandler builds a Handler with one model whose cached-read rate is
// far below its input rate, so cache-aware vs cache-blind pricing diverge
// measurably.
func cachedModelHandler() *Handler {
	cfg := &config.Config{}
	cfg.Providers.Anthropic = &config.ProviderConfig{Models: []config.ModelConfig{
		{ID: "sonnet", Tier: 2, InputPricePer1M: 3, CachedInputPricePer1M: 0.30, CacheWritePricePer1M: 3.75, OutputPricePer1M: 15},
	}}
	return &Handler{
		router:  router.NewRouter(cfg, nil),
		pricing: pricing.NewTable(cfg.Providers),
	}
}

// Pre-request estimation is cache-BLIND on purpose: before the call we do not
// know what the provider will serve from cache, so estimatedCost must price all
// input at the full rate via EstimateCost.
func TestEstimatedCostIsCacheBlind(t *testing.T) {
	h := cachedModelHandler()
	model, _ := h.router.FindModel("sonnet")
	// ~400 chars -> ~100 prompt tokens; max_tokens caps output at 10.
	maxTok := 10
	req := &types.ChatCompletionRequest{
		Messages:  []types.Message{{Role: "user", Content: mkContent(400)}},
		MaxTokens: &maxTok,
	}
	// Sanity: ContentString must yield the 400-char body so the token estimate is
	// the 100 we price against below.
	if got := len(req.Messages[0].ContentString()); got != 400 {
		t.Fatalf("content length = %d, want 400", got)
	}
	got := h.estimatedCost(req, model)
	// Cache-blind expectation: 100 input @ 3/1M + 10 output @ 15/1M.
	want := h.pricing.EstimateCost("sonnet", 100, 10)
	if got != want {
		t.Fatalf("estimatedCost = %.10f, want cache-blind EstimateCost %.10f", got, want)
	}
}

// Post-response accounting is cache-AWARE: costAndSavings must route through
// EstimateUsageCost, pricing cache-read tokens at the cached rate, so a mostly
// cached turn costs strictly less than the cache-blind estimate of the same
// token totals.
func TestCostAndSavingsIsCacheAware(t *testing.T) {
	h := cachedModelHandler()
	usage := types.Usage{
		PromptTokens:         100,
		CompletionTokens:     10,
		UncachedInputTokens:  20,
		CacheReadInputTokens: 80, // 80 of 100 input served from cache
	}
	cost, _ := h.costAndSavings("sonnet", usage)
	cacheBlind := h.pricing.EstimateCost("sonnet", 100, 10)
	if !(cost.TotalUSD < cacheBlind) {
		t.Fatalf("cache-aware cost %.10f must be below cache-blind %.10f", cost.TotalUSD, cacheBlind)
	}
	// And it must equal the explicit per-rate breakdown.
	want := h.pricing.EstimateUsageCost("sonnet", usage)
	if cost.TotalUSD != want.TotalUSD {
		t.Fatalf("costAndSavings total %.10f != EstimateUsageCost %.10f", cost.TotalUSD, want.TotalUSD)
	}
}

// mkContent returns a JSON string literal of n 'a' chars, so ContentString
// decodes it back to an n-char plain string.
func mkContent(n int) json.RawMessage {
	b := make([]byte, n+2)
	b[0] = '"'
	for i := 1; i <= n; i++ {
		b[i] = 'a'
	}
	b[n+1] = '"'
	return json.RawMessage(b)
}
