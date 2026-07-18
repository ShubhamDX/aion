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

	// Response-action governance (optional): when installed, the moment the FIRST
	// tool-call fragment appears the proxy stops emitting and buffers EVERY
	// subsequent chunk (text, usage, finish) in order through the terminal marker,
	// so the protocol order is preserved and no finish chunk races ahead of the
	// tool deltas. It evaluates the complete call set only after a clean EOF, then
	// replays the original chunk sequence for an all-allowed result, or emits one
	// ordinary completion for block/hold/overflow/error. A nil hook keeps the
	// original straight-through relay.
	governTools := h.hooks != nil && h.hooks.ResponseAction != nil
	var toolBuf *streamToolBuffer
	if governTools {
		toolBuf = newStreamToolBuffer()
	}
	buffering := false // set once the first tool-call fragment is seen

	writeChunk := func(chunk *types.ChatCompletionChunk) bool {
		data, marshalErr := json.Marshal(chunk)
		if marshalErr != nil {
			slog.Error("stream marshal error", "error", marshalErr)
			return false
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return true
	}

	readErr := false
	for {
		chunk, err := stream.ReadChunk()
		if err != nil {
			if err == io.EOF {
				streamComplete = true
				break
			}
			// Non-EOF read error: never evaluate or replay partial calls. If we were
			// already buffering a governed tool-call sequence, fail closed below.
			slog.Error("stream read error", "error", err)
			readErr = true
			break
		}

		// Accumulate usage if the provider sends it.
		if chunk.Usage != nil {
			totalUsage.MergeFrom(*chunk.Usage)
		}

		// Enter buffering at the first tool-call fragment and stay there.
		if governTools && !buffering && chunkHasToolCall(chunk) {
			buffering = true
		}

		if buffering {
			// Buffer the whole tail (tool deltas AND interleaved text/usage/finish)
			// in arrival order; nothing is released until governance finishes.
			toolBuf.add(chunk)
			continue
		}

		if !writeChunk(chunk) {
			break
		}
	}

	// If governance began buffering a tool-call sequence, resolve it now.
	if governTools && buffering {
		switch {
		case readErr:
			// Upstream failed mid-stream: the buffered calls may be incomplete. Fail
			// closed with a forced block; release no buffered fragment.
			writeChunk(failClosedEnvelopeChunk(model.ID, "upstream_stream_error"))
		default:
			proposed, perr := toolBuf.proposed()
			if perr != nil {
				// Overflowed a memory ceiling: fail closed, release nothing.
				writeChunk(failClosedEnvelopeChunk(model.ID, "tool_args_too_large"))
			} else {
				decision := h.hooks.ResponseAction(types.ResponseActionInput{
					RequestID:      requestID,
					PrincipalID:    keyIDFromInfo(keyInfo),
					RequestDigest:  types.RequestContentDigest(req),
					Protocol:       ingressProtocol(ctx),
					RoutedProvider: model.Provider,
					RoutedModel:    model.ID,
					Tier:           tier,
					ToolCalls:      proposed,
				})
				if decision.AllAllowedValidated(len(proposed)) {
					// Replay the ORIGINAL buffered sequence verbatim, preserving order
					// and any interleaved content.
					for _, c := range toolBuf.bufferedChunks {
						if !writeChunk(c) {
							break
						}
					}
				} else {
					writeChunk(envelopeChunk(model.ID, proposed, decision))
				}
			}
		}
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
