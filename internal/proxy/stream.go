package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/ShubhamDX/aion/internal/apikey"
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
) {
	ctx := r.Context()

	stream, err := prov.SendStream(ctx, req, model.ID)
	if err != nil {
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

	for {
		chunk, err := stream.ReadChunk()
		if err != nil {
			if err == io.EOF {
				break
			}
			slog.Error("stream read error", "error", err)
			break
		}

		// Accumulate usage if the provider sends it in the final chunk.
		if chunk.Usage != nil {
			totalUsage = *chunk.Usage
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
	if totalUsage.TotalTokens > 0 {
		costUSD = h.pricing.EstimateCost(
			model.ID,
			totalUsage.PromptTokens,
			totalUsage.CompletionTokens,
		)
		maxCost := h.pricing.MostExpensiveModelCost(
			totalUsage.PromptTokens,
			totalUsage.CompletionTokens,
		)
		savingsUSD = maxCost - costUSD
		if savingsUSD < 0 {
			savingsUSD = 0
		}
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

	// Record budget spend.
	if keyInfo != nil && h.budget != nil && costUSD > 0 {
		_ = h.budget.Record(ctx, keyInfo.Key, costUSD)
	}
}
