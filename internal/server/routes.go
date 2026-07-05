package server

import (
	"encoding/json"
	"net/http"
)

// RouteHandlers holds the handler functions for each endpoint.
type RouteHandlers struct {
	ChatCompletion    http.HandlerFunc // POST /v1/chat/completions
	Responses         http.HandlerFunc // POST /v1/responses
	AnthropicMessages http.HandlerFunc // POST /v1/messages
	ListModels        http.HandlerFunc // GET /v1/models
	Health            http.HandlerFunc // GET /health
	MetricsSavings    http.HandlerFunc // GET /aion/v1/metrics/savings
	MetricsRouting    http.HandlerFunc // GET /aion/v1/metrics/routing
	MetricsCosts      http.HandlerFunc // GET /aion/v1/metrics/costs
}

// NewRouter creates a new ServeMux with all routes registered using Go 1.22+
// method-prefixed pattern syntax.
func NewRouter(handlers RouteHandlers) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", handlers.ChatCompletion)
	mux.HandleFunc("POST /v1/responses", handlers.Responses)
	mux.HandleFunc("POST /v1/messages", handlers.AnthropicMessages)
	mux.HandleFunc("POST /v1/messages/count_tokens", countTokensStub)
	mux.HandleFunc("GET /v1/models", handlers.ListModels)
	mux.HandleFunc("GET /health", handlers.Health)
	mux.HandleFunc("GET /aion/v1/metrics/savings", handlers.MetricsSavings)
	mux.HandleFunc("GET /aion/v1/metrics/routing", handlers.MetricsRouting)
	mux.HandleFunc("GET /aion/v1/metrics/costs", handlers.MetricsCosts)
	return mux
}

// countTokensStub returns a minimal response for /v1/messages/count_tokens.
// Claude Code calls this endpoint for preflight token counting; returning a
// plausible estimate prevents 404 errors.
func countTokensStub(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int{"input_tokens": 0})
}
