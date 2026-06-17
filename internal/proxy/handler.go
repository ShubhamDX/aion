package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/server"
	"github.com/ShubhamDX/aion/internal/telemetry"
	"github.com/ShubhamDX/aion/internal/types"
	pkgclassifier "github.com/ShubhamDX/aion/pkg/classifier"
)

// sessionIDHeader is the optional client-supplied conversation id header. When
// present it is the highest-precedence session identity (see
// types.SessionMaterialFromRequest). It is read but never logged raw.
const sessionIDHeader = "X-AION-Session-Id"

// BudgetChecker validates whether a request is within budget and records spend.
type BudgetChecker interface {
	Check(ctx context.Context, apiKeyID string, dailyLimit, monthlyLimit float64) error
	Record(ctx context.Context, apiKeyID string, costUSD float64) error
}

// Handler orchestrates the full lifecycle of a proxied chat completion request:
// classify, route, budget-check, dispatch to provider, record telemetry.
type Handler struct {
	classifier pkgclassifier.Classifier
	router     *router.Router
	registry   *provider.Registry
	budget     BudgetChecker
	pricing    *pricing.Table
	recorder   *telemetry.Recorder
	// hooks is the optional gateway extension surface (an embedding product wraps
	// the request lifecycle here). nil leaves OSS behavior unchanged.
	hooks *types.GatewayHooks
	// postResponseWG tracks in-flight asynchronous PostResponse hooks so shutdown
	// can drain them before the embedding product closes its store. Without this a
	// graceful shutdown could close the evidence store under a running hook,
	// dropping the signed row (and racing the store).
	postResponseWG sync.WaitGroup
}

// SetGatewayHooks attaches the optional pre-request / post-response hooks. nil
// hooks (or nil func fields) are no-ops.
func (h *Handler) SetGatewayHooks(hooks *types.GatewayHooks) { h.hooks = hooks }

// NewHandler creates a new proxy Handler. budget and recorder may be nil; when
// nil the corresponding functionality is silently skipped.
func NewHandler(
	cls pkgclassifier.Classifier,
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

	case model == "aion-local":
		// Force routing to the local llama.cpp provider.
		tier = types.Tier1
		var err error
		selectedModel, err = h.router.FindByProvider("local")
		if err != nil {
			writeError(w, http.StatusServiceUnavailable, "no_model_available",
				"No local model available: "+err.Error())
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

	// 2b. Gateway pre-request hook (optional). An embedding product runs its
	// decision here: block, hold for approval, or override the routed model
	// (cheaper-safe routing). A nil hook leaves behavior unchanged.
	sessionMaterial := types.SessionMaterialFromRequest(&req, r.Header.Get(sessionIDHeader))
	// Material extracted; scrub the raw session id so it never reaches the hook
	// (digests only) or an upstream provider.
	types.ScrubSessionID(&req)
	if h.hooks != nil && h.hooks.PreRequest != nil {
		estCost := h.estimatedCost(&req, selectedModel)
		dec := h.hooks.PreRequest(types.PreRequestInput{
			RequestID:       requestID,
			PrincipalID:     keyIDFromInfo(keyInfo),
			Request:         &req,
			RequestedModel:  model,
			RoutedModel:     selectedModel.ID,
			RoutedProvider:  selectedModel.Provider,
			Tier:            tier,
			EstimatedCost:   estCost,
			RequestDigest:   types.RequestContentDigest(&req),
			SessionMaterial: sessionMaterial,
		})
		switch dec.Verdict {
		case types.VerdictBlock:
			w.Header().Set("X-AION-Decision", "block")
			writeError(w, http.StatusForbidden, "policy_block", decisionMessage(dec, "request blocked by policy"))
			return
		case types.VerdictHold:
			w.Header().Set("X-AION-Decision", "hold")
			writeError(w, http.StatusAccepted, "approval_required", decisionMessage(dec, "request held for approval"))
			return
		case types.VerdictRoute:
			m, changed, err := h.resolveRouteOverride(dec, selectedModel)
			if err != nil {
				// Fail closed: a policy that ordered an unresolvable route cannot
				// be silently ignored (that would let the request proceed on the
				// un-governed original route).
				w.Header().Set("X-AION-Decision", "route_error")
				writeError(w, http.StatusInternalServerError, "route_override_failed", err.Error())
				return
			}
			if changed {
				selectedModel = m
				tier = m.Tier
				w.Header().Set("X-AION-Decision", "route")
			}
		}
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
	w.Header().Set("X-AION-Provider", selectedModel.Provider)
	w.Header().Set("X-Request-ID", requestID)

	// 6. Dispatch -- streaming or non-streaming.
	if req.Stream {
		h.handleStream(w, r, &req, prov, selectedModel, tier, score, signals, keyInfo, requestID, start, sessionMaterial)
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
	var costBreakdown pricing.CostBreakdown
	if resp.ChatResponse != nil && resp.ChatResponse.Usage.TotalTokens > 0 {
		costBreakdown, savingsUSD = h.costAndSavings(selectedModel.ID, resp.ChatResponse.Usage)
		costUSD = costBreakdown.TotalUSD
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

	// Write response FIRST, so the customer response is fully served before the
	// post-response hook runs. The hook records signed evidence (and an embedding
	// product may validate the response shape there); doing that work before the
	// write would let it delay or, on a panic, block the served response. The
	// contract is that PostResponse observes a response the client already has.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	json.NewEncoder(w).Encode(resp.ChatResponse)

	// Gateway post-response hook (optional): the embedding product records the
	// signed evidence row for this completed request. Dispatched ASYNCHRONOUSLY
	// (after the response is written) so the hook's work cannot delay the served
	// response, even via net/http response buffering before handler return.
	if h.hooks != nil && h.hooks.PostResponse != nil && resp.ChatResponse != nil {
		// Non-stream: the full response is available, so compute the next turn's
		// prefix digest now.
		postMaterial := sessionMaterial
		postMaterial.NextCachePrefixMaterialSHA256 = types.NextCachePrefixMaterial(&req, resp.ChatResponse)
		h.dispatchPostResponse(postResponseInputWithUsage(types.PostResponseInput{
			RequestID:        requestID,
			PrincipalID:      keyIDFromInfo(keyInfo),
			RequestedModel:   model,
			RoutedModel:      selectedModel.ID,
			RoutedProvider:   selectedModel.Provider,
			Tier:             tier,
			InputTokens:      resp.ChatResponse.Usage.PromptTokens,
			OutputTokens:     resp.ChatResponse.Usage.CompletionTokens,
			CostUSD:          costUSD,
			SavingsUSD:       savingsUSD,
			LatencyMS:        time.Since(start).Milliseconds(),
			StatusCode:       resp.StatusCode,
			Stream:           false,
			RequestDigest:    types.RequestContentDigest(&req),
			ResponseDigest:   types.ResponseContentDigest(resp.ChatResponse),
			SessionMaterial:  postMaterial,
			ResponseContents: types.ResponseContentStrings(resp.ChatResponse),
		}, resp.ChatResponse.Usage, costBreakdown))
	}
}

// estimatedCost estimates the cost of a request against a model before dispatch,
// using a rough token estimate from the prompt. Used to feed the pre-request
// budget decision; the post-response path records actual cost.
func (h *Handler) estimatedCost(req *types.ChatCompletionRequest, model *router.ModelOption) float64 {
	if h.pricing == nil || model == nil {
		return 0
	}
	// Rough prompt-token estimate: ~4 chars/token across all message content.
	chars := 0
	for _, m := range req.Messages {
		chars += len(m.ContentString())
	}
	promptTokens := chars / 4
	outTokens := 256
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		outTokens = *req.MaxTokens
	}
	return h.pricing.EstimateCost(model.ID, promptTokens, outTokens)
}

// resolveRouteOverride resolves a VerdictRoute decision to a concrete model.
// Precedence: an explicit RoutedModelOverride (pinned model id) wins; otherwise
// a RoutedTierOverride routes to the cheapest healthy model in EXACTLY that tier
// from THIS deployment's catalog (strict, no tier fallback). Both forms are
// constrained to RoutedAllowedProviders when set. It returns the resolved
// model, whether the route actually changed the selection, and an error that
// the caller MUST treat as fail-closed (a policy that ordered an unresolvable
// route cannot fall through to the original un-governed model). A decision with
// neither field set returns (current, false, nil): a route verdict that names
// no target is a no-op, not an error.
func (h *Handler) resolveRouteOverride(dec types.PreRequestDecision, current *router.ModelOption) (*router.ModelOption, bool, error) {
	// RoutedAllowedProviders constrains BOTH override forms: a pinned model and a
	// tier target must land on an allowed provider. Empty means no constraint.
	var allowed map[string]bool
	if len(dec.RoutedAllowedProviders) > 0 {
		allowed = make(map[string]bool, len(dec.RoutedAllowedProviders))
		for _, p := range dec.RoutedAllowedProviders {
			allowed[p] = true
		}
	}

	if dec.RoutedModelOverride != "" {
		// Resolve the pinned model THROUGH the allowlist: a policy that pinned a
		// model from a disallowed provider, or a model id that maps to more than
		// one provider, fails closed rather than silently sending traffic off the
		// allowed set. This closes the pinned-route provider-boundary gap.
		m, err := h.router.FindModelConstrained(dec.RoutedModelOverride, allowed)
		if err != nil {
			return nil, false, fmt.Errorf("policy route to model %q: %w", dec.RoutedModelOverride, err)
		}
		if m.ID == current.ID && m.Provider == current.Provider {
			return current, false, nil
		}
		return m, true, nil
	}
	if dec.RoutedTierOverride > 0 {
		tier := types.Tier(dec.RoutedTierOverride)
		// Strict: select within EXACTLY this tier, constrained to the policy's
		// allowed providers. No tier fallback and no provider widening, so a
		// downgrade that names tier N (and a provider set) cannot silently land
		// on another tier or a disallowed provider.
		m, err := h.router.RouteInTier(tier, allowed)
		if err != nil {
			return nil, false, fmt.Errorf("policy routed to tier %d with no eligible healthy model: %w", dec.RoutedTierOverride, err)
		}
		if m.ID == current.ID {
			return current, false, nil
		}
		return m, true, nil
	}
	return current, false, nil
}

// decisionMessage returns the decision's client-facing message, or a default.
func decisionMessage(d types.PreRequestDecision, def string) string {
	if d.Message != "" {
		return d.Message
	}
	return def
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
