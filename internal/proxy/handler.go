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

// BudgetChecker reserves worst-case request cost and settles actual spend.
type BudgetChecker interface {
	Reserve(ctx context.Context, apiKeyID string, estimate, dailyLimit, monthlyLimit float64) (string, error)
	Settle(ctx context.Context, apiKeyID, date string, estimate, actual float64) error
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
		// Carry any schema settings the PreRequest decision resolved (SP2b-2). The
		// post-route seam below supersedes this when present; it is kept so a hook
		// that resolves at PreRequest time still works. `json:"-"`, never upstream.
		if dec.SchemaSettings != nil {
			req.SchemaSettings = dec.SchemaSettings
		}
	}

	// 2c. Post-route schema-settings seam (SP2b-3a). Runs AFTER any route override,
	// so selectedModel is the FINAL route, but BEFORE dispatch. The embedding
	// product resolves schema settings against the final provider/model. The
	// result is stamped onto the request's json:"-" carrier; no provider reads it,
	// so it cannot change a served response or an upstream payload.
	if h.hooks != nil && h.hooks.ResolveSchemaSettings != nil {
		req.SchemaSettings = h.hooks.ResolveSchemaSettings(types.PostRouteInput{
			RequestID:      requestID,
			PrincipalID:    keyIDFromInfo(keyInfo),
			RequestedModel: model,
			RoutedProvider: selectedModel.Provider,
			RoutedModel:    selectedModel.ID,
			Tier:           tier,
			RequestDigest:  types.RequestContentDigest(&req),
		})
	}

	// 3. Resolve provider.
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

	// 5b. Schema fail-closed gate (SP2b-3b): a mandatory schema policy whose native
	// mode cannot be honored on the FINAL route returns a schema error BEFORE
	// dispatch, rather than silently serving an unconstrained response. Scoped to
	// the schema-settings carrier the post-route seam set; only mandatory
	// fail-closed acts (an optional fallback dispatches normally).
	if schemaFailClosed(&req) {
		w.Header().Set("X-AION-Decision", "schema_fail_closed")
		writeError(w, http.StatusUnprocessableEntity, "schema_policy_unsatisfiable",
			"schema policy requires structured output the routed model cannot honor")
		return
	}

	// 6. Dispatch -- streaming or non-streaming.
	if req.Stream {
		reservationDate, reservedCost, err := h.reserveBudget(ctx, &req, selectedModel, keyInfo)
		if err != nil {
			writeError(w, http.StatusTooManyRequests, "budget_exceeded", err.Error())
			return
		}
		h.handleStream(w, r, &req, prov, selectedModel, tier, score, signals, keyInfo, requestID, start, sessionMaterial, reservationDate, reservedCost)
		return
	}

	// 5c. Context-compression seam (CP4): non-stream only, after routing + schema
	// and BEFORE the budget reservation below, so the reservation estimates from
	// the payload actually dispatched. The embedding product returns a
	// deterministically compressed message set (computed from the ORIGINAL
	// intent); the proxy swaps req.Messages just before building the upstream
	// payload. Routing already used the original request, so this cannot change
	// the routed model.
	if h.hooks != nil && h.hooks.ApplyContextCompression != nil {
		if res := h.hooks.ApplyContextCompression(types.PostRouteInput{
			RequestID:      requestID,
			PrincipalID:    keyIDFromInfo(keyInfo),
			RequestedModel: model,
			RoutedProvider: selectedModel.Provider,
			RoutedModel:    selectedModel.ID,
			Tier:           tier,
			RequestDigest:  types.RequestContentDigest(&req),
		}); res != nil && res.Messages != nil {
			req.Messages = res.Messages
		}
	}

	// 5d. Output-control seam (OP3b): non-stream only, after context compression
	// and before dispatch. Nil hook or nil result leaves the request unchanged.
	h.applyOutputControl(&req, requestID, keyInfo, model, selectedModel, tier)

	reservationDate, reservedCost, err := h.reserveBudget(ctx, &req, selectedModel, keyInfo)
	if err != nil {
		writeError(w, http.StatusTooManyRequests, "budget_exceeded", err.Error())
		return
	}

	// Non-streaming path.
	resp, err := prov.Send(ctx, &req, selectedModel.ID)
	if err != nil {
		h.settleBudget(ctx, keyInfo, reservationDate, reservedCost, reservedCost)
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

	settledCost := reservedCost
	if resp.ChatResponse != nil && resp.ChatResponse.Usage.TotalTokens > 0 {
		settledCost = costUSD
	}
	h.settleBudget(ctx, keyInfo, reservationDate, reservedCost, settledCost)

	// Response-action governance (optional): normalize the complete proposed tool
	// calls and ask the hook to allow/block/hold BEFORE releasing them. If not all
	// are allowed, the response is rewritten to the action envelope so no
	// executable tool call reaches the client. Runs synchronously, before the
	// write, precisely because it can change the served bytes (unlike
	// PostResponse). A nil hook leaves the response unchanged.
	if h.hooks != nil && h.hooks.ResponseAction != nil && resp.ChatResponse != nil {
		proposed := proposedToolCallsFromResponse(resp.ChatResponse)
		if decision, applies := evaluateResponseAction(h.hooks.ResponseAction, types.ResponseActionInput{
			RequestID:      requestID,
			PrincipalID:    keyIDFromInfo(keyInfo),
			RequestDigest:  types.RequestContentDigest(&req),
			Protocol:       ingressProtocol(ctx),
			RoutedProvider: selectedModel.Provider,
			RoutedModel:    selectedModel.ID,
			Tier:           tier,
			ToolCalls:      proposed,
		}); applies && !decision.AllAllowedValidated(len(proposed)) {
			// Fail closed: any malformed decision (wrong cardinality, duplicate /
			// missing / out-of-range index, unknown verdict) or any non-allow verdict
			// rewrites to the envelope, so no unclassified tool call is released.
			rewriteResponseWithEnvelope(resp.ChatResponse, proposed, decision)
		}
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
			RequestID:           requestID,
			PrincipalID:         keyIDFromInfo(keyInfo),
			RequestedModel:      model,
			RoutedModel:         selectedModel.ID,
			RoutedProvider:      selectedModel.Provider,
			Tier:                tier,
			InputTokens:         resp.ChatResponse.Usage.PromptTokens,
			OutputTokens:        resp.ChatResponse.Usage.CompletionTokens,
			CostUSD:             costUSD,
			SavingsUSD:          savingsUSD,
			LatencyMS:           time.Since(start).Milliseconds(),
			StatusCode:          resp.StatusCode,
			Stream:              false,
			RequestDigest:       types.RequestContentDigest(&req),
			ResponseDigest:      types.ResponseContentDigest(resp.ChatResponse),
			SessionMaterial:     postMaterial,
			ResponseContents:    types.ResponseContentStrings(resp.ChatResponse),
			SchemaNativeEmitted: resp.SchemaNativeEmitted,
		}, resp.ChatResponse.Usage, costBreakdown))
	}
}

// estimatedCost computes a conservative reservation before dispatch. JSON byte
// length is used as an input-token upper bound, with framing allowance. Output
// uses the requested cap, the configured model cap or the provider default.
func (h *Handler) estimatedCost(req *types.ChatCompletionRequest, model *router.ModelOption) float64 {
	if h.pricing == nil || model == nil {
		return 0
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return 0
	}
	promptTokens := len(payload) + 64*len(req.Messages) + 256
	outTokens := model.MaxTokens
	if outTokens <= 0 {
		outTokens = 4096
	}
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		outTokens = *req.MaxTokens
	}
	return h.pricing.EstimateCost(model.ID, promptTokens, outTokens)
}

func (h *Handler) reserveBudget(
	ctx context.Context,
	req *types.ChatCompletionRequest,
	model *router.ModelOption,
	keyInfo *apikey.KeyInfo,
) (string, float64, error) {
	if keyInfo == nil || h.budget == nil {
		return "", 0, nil
	}
	estimate := h.estimatedCost(req, model)
	if (keyInfo.DailyLimitUSD > 0 || keyInfo.MonthlyLimitUSD > 0) &&
		model != nil && model.Provider != "local" && estimate <= 0 {
		return "", 0, fmt.Errorf("cost estimate unavailable for provider %q model %q", model.Provider, model.ID)
	}
	date, err := h.budget.Reserve(
		ctx, keyInfo.Key, estimate, keyInfo.DailyLimitUSD, keyInfo.MonthlyLimitUSD,
	)
	if err != nil {
		return "", 0, err
	}
	return date, estimate, nil
}

func (h *Handler) settleBudget(
	ctx context.Context,
	keyInfo *apikey.KeyInfo,
	date string,
	estimate float64,
	actual float64,
) {
	if keyInfo == nil || h.budget == nil || date == "" {
		return
	}
	if err := h.budget.Settle(context.WithoutCancel(ctx), keyInfo.Key, date, estimate, actual); err != nil {
		slog.Error("budget settlement failed", "api_key_id", keyIDFromInfo(keyInfo), "error", err)
	}
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

// schemaFailClosed reports whether a request must be refused before dispatch
// because a mandatory schema policy cannot be honored on the final route. Two
// cases, both scoped to the schema-settings carrier (requests without a schema
// policy are never affected):
//
//  1. The post-route resolver already flagged FailClosed (mandatory mode the
//     final route cannot honor at all).
//  2. Native output is MANDATORY (MustEmitNative) but cannot actually be emitted
//     on the wire because the caller already set response_format, which the
//     provider will not override. Serving anyway would silently bypass a
//     fail-closed provider-native policy with an unconstrained response.
//  3. Native output is MANDATORY but the request is streaming. A stream can emit
//     native output, but AION does not reassemble the stream, so the schema row
//     cannot validate it: there would be native emission with no signed evidence.
//     SP2b-3b refuses mandatory native on streams (signed streaming evidence is a
//     later slice). Observe/optional streams are unaffected.
//
// An advisory fallback (FailClosed=false, MustEmitNative=false) dispatches.
func schemaFailClosed(req *types.ChatCompletionRequest) bool {
	ss := req.SchemaSettings
	if ss == nil {
		return false
	}
	if ss.FailClosed {
		return true
	}
	if ss.MustEmitNative && req.ResponseFormat != nil {
		// Mandatory native, but the caller's response_format blocks emission.
		return true
	}
	if ss.MustEmitNative && req.Stream {
		// Mandatory native on a stream: emission possible but unverifiable.
		return true
	}
	return false
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
