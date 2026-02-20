package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/classifier"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/server"
	"github.com/ShubhamDX/aion/internal/telemetry"
	"github.com/ShubhamDX/aion/internal/types"
)

// BudgetChecker validates whether a request is within budget and records spend.
type BudgetChecker interface {
	Check(ctx context.Context, apiKeyID string, dailyLimit, monthlyLimit float64) error
	Record(ctx context.Context, apiKeyID string, costUSD float64) error
}

// Handler orchestrates the full lifecycle of a proxied chat completion request:
// classify, route, budget-check, dispatch to provider, record telemetry.
type Handler struct {
	classifier *classifier.Classifier
	router     *router.Router
	registry   *provider.Registry
	budget     BudgetChecker
	pricing    *pricing.Table
	recorder   *telemetry.Recorder
}

// NewHandler creates a new proxy Handler. budget and recorder may be nil; when
// nil the corresponding functionality is silently skipped.
func NewHandler(
	cls *classifier.Classifier,
	rtr *router.Router,
	reg *provider.Registry,
	bgt BudgetChecker,
	prc *pricing.Table,
	rec *telemetry.Recorder,
) *Handler {
	return &Handler{
		classifier: cls,
		router:     rtr,
		registry:   reg,
		budget:     bgt,
		pricing:    prc,
		recorder:   rec,
	}
}

// ChatCompletion handles POST /v1/chat/completions.
func (h *Handler) ChatCompletion(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()
	requestID := server.RequestID(ctx)
	keyInfo := server.APIKeyInfo(ctx)

	// 1. Parse request body.
	var req types.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "Failed to parse request body: "+err.Error())
		return
	}

	// 2. Determine routing.
	var (
		selectedModel *router.ModelOption
		tier          types.Tier
		score         float64
		signals       map[string]float64
	)

	model := req.Model
	switch {
	case model == "" || model == "aion-auto":
		// Classify complexity and pick the cheapest suitable model.
		tier, score, signals = h.classifier.Classify(&req)
		var err error
		selectedModel, err = h.router.RouteWithFallback(tier)
		if err != nil {
			writeError(w, http.StatusServiceUnavailable, "no_model_available",
				"No healthy model available for tier: "+err.Error())
			return
		}

	case model == "aion-escalate":
		// Force highest tier.
		tier = types.Tier3
		var err error
		selectedModel, err = h.router.RouteWithFallback(types.Tier3)
		if err != nil {
			writeError(w, http.StatusServiceUnavailable, "no_model_available", err.Error())
			return
		}

	default:
		// Specific model requested -- bypass classification.
		var err error
		selectedModel, err = h.router.FindModel(model)
		if err != nil {
			writeError(w, http.StatusBadRequest, "model_not_found", "Model not found: "+model)
			return
		}
		tier = selectedModel.Tier
	}

	// 3. Budget check.
	if keyInfo != nil && h.budget != nil {
		if err := h.budget.Check(ctx, keyInfo.Key, keyInfo.DailyLimitUSD, keyInfo.MonthlyLimitUSD); err != nil {
			writeError(w, http.StatusTooManyRequests, "budget_exceeded", err.Error())
			return
		}
	}

	// 4. Resolve provider.
	prov, ok := h.registry.Get(selectedModel.Provider)
	if !ok {
		writeError(w, http.StatusInternalServerError, "provider_error",
			"Provider not found: "+selectedModel.Provider)
		return
	}

	// 5. Log routing decision and set AION response headers.
	slog.Info("routed",
		"request_id", requestID,
		"ingress", "openai",
		"requested_model", model,
		"routed_model", selectedModel.ID,
		"provider", selectedModel.Provider,
		"tier", int(tier),
		"stream", req.Stream,
	)
	w.Header().Set("X-AION-Tier", fmt.Sprintf("%d", int(tier)))
	w.Header().Set("X-AION-Model", selectedModel.ID)
	w.Header().Set("X-Request-ID", requestID)

	// 6. Dispatch -- streaming or non-streaming.
	if req.Stream {
		h.handleStream(w, r, &req, prov, selectedModel, tier, score, signals, keyInfo, requestID, start)
		return
	}

	// Non-streaming path.
	resp, err := prov.Send(ctx, &req, selectedModel.ID)
	if err != nil {
		slog.Error("provider error",
			"provider", selectedModel.Provider,
			"model", selectedModel.ID,
			"error", err,
		)
		writeError(w, http.StatusBadGateway, "provider_error",
			"Provider request failed: "+err.Error())
		return
	}

	// Calculate cost and savings.
	var costUSD, savingsUSD float64
	if resp.ChatResponse != nil && resp.ChatResponse.Usage.TotalTokens > 0 {
		costUSD = h.pricing.EstimateCost(
			selectedModel.ID,
			resp.ChatResponse.Usage.PromptTokens,
			resp.ChatResponse.Usage.CompletionTokens,
		)
		maxCost := h.pricing.MostExpensiveModelCost(
			resp.ChatResponse.Usage.PromptTokens,
			resp.ChatResponse.Usage.CompletionTokens,
		)
		savingsUSD = maxCost - costUSD
		if savingsUSD < 0 {
			savingsUSD = 0
		}
	}

	w.Header().Set("X-AION-Cost-USD", fmt.Sprintf("%.6f", costUSD))
	w.Header().Set("X-AION-Savings-USD", fmt.Sprintf("%.6f", savingsUSD))

	// Record telemetry asynchronously.
	if h.recorder != nil && resp.ChatResponse != nil {
		h.recorder.Record(telemetry.RequestEvent{
			RequestID:    requestID,
			APIKeyID:     keyIDFromInfo(keyInfo),
			Tier:         int(tier),
			Model:        selectedModel.ID,
			Provider:     selectedModel.Provider,
			InputTokens:  resp.ChatResponse.Usage.PromptTokens,
			OutputTokens: resp.ChatResponse.Usage.CompletionTokens,
			CostUSD:      costUSD,
			SavingsUSD:   savingsUSD,
			LatencyMS:    time.Since(start).Milliseconds(),
			StatusCode:   resp.StatusCode,
			Stream:       false,
			CreatedAt:    time.Now().UTC(),
		})
	}

	// Record budget spend.
	if keyInfo != nil && h.budget != nil && costUSD > 0 {
		_ = h.budget.Record(ctx, keyInfo.Key, costUSD)
	}

	// Write response.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	json.NewEncoder(w).Encode(resp.ChatResponse)
}

// writeError writes an OpenAI-compatible JSON error response.
func writeError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(types.ErrorResponse{
		Error: types.ErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}

// keyIDFromInfo returns a stable identifier for telemetry. Returns "anonymous"
// when no key info is available.
func keyIDFromInfo(info *apikey.KeyInfo) string {
	if info == nil {
		return "anonymous"
	}
	return info.Name
}
