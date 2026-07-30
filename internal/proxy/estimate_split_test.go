package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ShubhamDX/aion/internal/budget"
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
		{ID: "sonnet", Tier: 2, InputPricePer1M: 3, CachedInputPricePer1M: 0.30, CacheWritePricePer1M: 3.75, OutputPricePer1M: 15, MaxTokens: 4096},
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
	// max_tokens caps output at 10.
	maxTok := 10
	req := &types.ChatCompletionRequest{
		Messages:  []types.Message{{Role: "user", Content: mkContent(400)}},
		MaxTokens: &maxTok,
	}
	got := h.estimatedCost(req, model)
	payload, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	// The reservation prices every serialized byte plus framing allowance at
	// the full input rate. It does not assume a future cache hit.
	want := h.pricing.EstimateCost("sonnet", len(payload)+64*len(req.Messages)+256, 10)
	if got != want {
		t.Fatalf("estimatedCost = %.10f, want cache-blind EstimateCost %.10f", got, want)
	}
}

// When the request omits max_tokens, the reservation uses a realistic
// multiple of the prompt (defaultOutputTokenRatio), not the model's full
// output ceiling: assuming the ceiling on every request reserves far more
// budget headroom than most turns use.
func TestEstimatedCostUsesRatioHeuristicWhenRequestOmitsMaxTokens(t *testing.T) {
	h := cachedModelHandler()
	model, _ := h.router.FindModel("sonnet")
	req := &types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: mkContent(20)}}}
	payload, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	promptTokens := len(payload) + 64*len(req.Messages) + 256
	wantOutTokens := int(float64(promptTokens) * defaultOutputTokenRatio)
	if wantOutTokens >= model.MaxTokens {
		t.Fatalf("test fixture must keep the ratio estimate (%d) below the model cap (%d) to exercise the uncapped path", wantOutTokens, model.MaxTokens)
	}
	want := h.pricing.EstimateCost("sonnet", promptTokens, wantOutTokens)
	if got := h.estimatedCost(req, model); got != want {
		t.Fatalf("estimatedCost = %.10f, want ratio-heuristic estimate %.10f", got, want)
	}
}

// A large enough prompt pushes the ratio heuristic past the model's
// configured output cap; the reservation must clamp to the cap rather than
// reserve more than the model can ever emit.
func TestEstimatedCostClampsRatioEstimateToModelMaxTokens(t *testing.T) {
	h := cachedModelHandler()
	model, _ := h.router.FindModel("sonnet")
	req := &types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: mkContent(6000)}}}
	payload, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	promptTokens := len(payload) + 64*len(req.Messages) + 256
	if int(float64(promptTokens)*defaultOutputTokenRatio) <= model.MaxTokens {
		t.Fatalf("test fixture must push the ratio estimate above the model cap (%d) to exercise the clamp", model.MaxTokens)
	}
	want := h.pricing.EstimateCost("sonnet", promptTokens, model.MaxTokens)
	if got := h.estimatedCost(req, model); got != want {
		t.Fatalf("estimatedCost = %.10f, want cap-clamped estimate %.10f", got, want)
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

// writeBudgetError is what actually reaches the client on a budget block; this
// verifies the real HTTP output (status, headers, JSON body), not just the
// ExceededError string formatting in the budget package.
func TestWriteBudgetErrorSetsRetryAfterAndCustomerMessage(t *testing.T) {
	exceeded := &budget.ExceededError{
		Scope: "monthly", Used: 3.7708, Estimate: 1.2798, Limit: 5,
		ResetAt: time.Now().UTC().Add(2 * time.Hour).Truncate(time.Second),
	}
	w := httptest.NewRecorder()
	writeBudgetError(w, exceeded)

	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusTooManyRequests)
	}
	wantRetryAfter := strconv.Itoa(int(time.Until(exceeded.ResetAt).Round(time.Second).Seconds()))
	if got := w.Header().Get("Retry-After"); got != wantRetryAfter {
		t.Fatalf("Retry-After = %q, want %q", got, wantRetryAfter)
	}
	if got, want := w.Header().Get("X-AION-Budget-Reset"), exceeded.ResetAt.Format(time.RFC3339); got != want {
		t.Fatalf("X-AION-Budget-Reset = %q, want %q", got, want)
	}
	var body types.ErrorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("response body is not valid JSON: %v (%s)", err, w.Body.String())
	}
	if body.Error.Type != "budget_exceeded" {
		t.Fatalf("error.type = %q, want %q", body.Error.Type, "budget_exceeded")
	}
	if strings.Contains(body.Error.Message, "reserved") {
		t.Fatalf("error.message %q still leaks internal reservation jargon", body.Error.Message)
	}
	if got, want := body.Error.Message, exceeded.CustomerMessage(); got != want {
		t.Fatalf("error.message = %q, want %q", got, want)
	}
	t.Logf("actual 429 response: status=%d Retry-After=%s X-AION-Budget-Reset=%s body=%s",
		w.Code, w.Header().Get("Retry-After"), w.Header().Get("X-AION-Budget-Reset"), w.Body.String())
}

// A non-ExceededError (a storage failure, not an over-budget request) still
// renders as a 429 but without the retry/reset headers, since there is no
// reset time to report.
func TestWriteBudgetErrorFallsBackForNonExceededError(t *testing.T) {
	w := httptest.NewRecorder()
	writeBudgetError(w, fmt.Errorf("budget: reserve usage: storage unavailable"))

	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusTooManyRequests)
	}
	if got := w.Header().Get("Retry-After"); got != "" {
		t.Fatalf("Retry-After = %q, want empty for a non-budget-exceeded error", got)
	}
	var body types.ErrorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("response body is not valid JSON: %v (%s)", err, w.Body.String())
	}
	if body.Error.Message != "budget: reserve usage: storage unavailable" {
		t.Fatalf("error.message = %q, want the raw storage error", body.Error.Message)
	}
}
