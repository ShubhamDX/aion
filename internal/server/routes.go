package server

import "net/http"

// RouteHandlers holds the handler functions for each endpoint.
type RouteHandlers struct {
	ChatCompletion http.HandlerFunc // POST /v1/chat/completions
	ListModels     http.HandlerFunc // GET /v1/models
	Health         http.HandlerFunc // GET /health
	MetricsSavings http.HandlerFunc // GET /aion/v1/metrics/savings
	MetricsRouting http.HandlerFunc // GET /aion/v1/metrics/routing
	MetricsCosts   http.HandlerFunc // GET /aion/v1/metrics/costs
}

// NewRouter creates a new ServeMux with all routes registered using Go 1.22+
// method-prefixed pattern syntax.
func NewRouter(handlers RouteHandlers) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", handlers.ChatCompletion)
	mux.HandleFunc("GET /v1/models", handlers.ListModels)
	mux.HandleFunc("GET /health", handlers.Health)
	mux.HandleFunc("GET /aion/v1/metrics/savings", handlers.MetricsSavings)
	mux.HandleFunc("GET /aion/v1/metrics/routing", handlers.MetricsRouting)
	mux.HandleFunc("GET /aion/v1/metrics/costs", handlers.MetricsCosts)
	return mux
}
