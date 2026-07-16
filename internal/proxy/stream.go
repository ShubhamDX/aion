package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/telemetry"
	"github.com/ShubhamDX/aion/internal/types"
)

// handleStream dispatches a streaming chat completion request and relays
// server-sent events (SSE) back to the client.
func (h *Handler) handleStream(
	w http.ResponseWriter,
	r *http.Request,
	req *types.ChatCompletionRequest,
	prov provider.Provider,
	model *router.ModelOption,
	tier types.Tier,
	score float64,
	signals map[string]float64,
	keyInfo *apikey.KeyInfo,
	requestID string,
	start time.Time,
	sessionMaterial types.SessionMaterial,
	reservationDate string,
	reservedCost float64,
) {
	ctx := r.Context()

	stream, err := prov.SendStream(ctx, req, model.ID)
	if err != nil {
		h.settleBudget(ctx, keyInfo, reservationDate, reservedCost, reservedCost)
		slog.Error("stream error",
			"provider", model.Provider,
			"model", model.ID,
			"error", err,
		)
		writeError(w, http.StatusBadGateway, "provider_error",
			"Stream request failed: "+err.Error())
		return
	}
	defer stream.Close()

	// Verify the ResponseWriter supports flushing.
	flusher, ok := w.(http.Flusher)
	if !ok {
		h.settleBudget(ctx, keyInfo, reservationDate, reservedCost, reservedCost)
		writeError(w, http.StatusInternalServerError, "server_error",
			"Streaming not supported")
		return
	}

	// Set SSE headers before writing the first byte.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	var totalUsage types.Usage
	streamComplete := false

	for {
		chunk, err := stream.ReadChunk()
		if err != nil {
			if err == io.EOF {
				streamComplete = true
				break
			}
			slog.Error("stream read error", "error", err)
			break
		}

		// Accumulate usage if the provider sends it.
		if chunk.Usage != nil {
			totalUsage.MergeFrom(*chunk.Usage)
		}

		data, marshalErr := json.Marshal(chunk)
		if marshalErr != nil {
			slog.Error("stream marshal error", "error", marshalErr)
			break
		}

		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Terminate the SSE stream.
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	// Calculate cost and savings from accumulated usage.
	var costUSD, savingsUSD float64
	var costBreakdown pricing.CostBreakdown
	if totalUsage.TotalTokens > 0 {
		costBreakdown, savingsUSD = h.costAndSavings(model.ID, totalUsage)
		costUSD = costBreakdown.TotalUSD
	}

	// Record telemetry asynchronously.
	if h.recorder != nil {
		h.recorder.Record(telemetry.RequestEvent{
			RequestID:    requestID,
			APIKeyID:     keyIDFromInfo(keyInfo),
			Tier:         int(tier),
			Model:        model.ID,
			Provider:     model.Provider,
			InputTokens:  totalUsage.PromptTokens,
			OutputTokens: totalUsage.CompletionTokens,
			CostUSD:      costUSD,
			SavingsUSD:   savingsUSD,
			LatencyMS:    time.Since(start).Milliseconds(),
			StatusCode:   http.StatusOK,
			Stream:       true,
			CreatedAt:    time.Now().UTC(),
		})
	}

	settledCost := reservedCost
	if streamComplete && totalUsage.TotalTokens > 0 {
		settledCost = costUSD
	}
	h.settleBudget(ctx, keyInfo, reservationDate, reservedCost, settledCost)

	// Gateway post-response hook (optional). Dispatched ASYNCHRONOUSLY (same
	// contract as the non-stream paths): the hook cannot delay stream completion,
	// a hook panic cannot crash the proxy, and DrainPostResponse tracks it.
	if h.hooks != nil && h.hooks.PostResponse != nil {
		// Streaming: the response body is not reassembled, so the next-turn prefix
		// digest stays "" (we do NOT buffer the stream to compute it) and
		// ResponseContents stays nil (an embedding product safe-degrades, e.g.
		// schema observe writes no row on a stream). This turn's session + prefix
		// material is still available from the request.
		h.dispatchPostResponse(postResponseInputWithUsage(types.PostResponseInput{
			RequestID:       requestID,
			PrincipalID:     keyIDFromInfo(keyInfo),
			RequestedModel:  req.Model,
			RoutedModel:     model.ID,
			RoutedProvider:  model.Provider,
			Tier:            tier,
			InputTokens:     totalUsage.PromptTokens,
			OutputTokens:    totalUsage.CompletionTokens,
			CostUSD:         costUSD,
			SavingsUSD:      savingsUSD,
			LatencyMS:       time.Since(start).Milliseconds(),
			StatusCode:      http.StatusOK,
			Stream:          true,
			RequestDigest:   types.RequestContentDigest(req),
			ResponseDigest:  "",
			SessionMaterial: sessionMaterial,
		}, totalUsage, costBreakdown))
	}
}
