package main

import (
	"context"
	"encoding/json"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/budget"
	"github.com/ShubhamDX/aion/internal/classifier"
	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/proxy"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/server"
	"github.com/ShubhamDX/aion/internal/telemetry"
	"github.com/ShubhamDX/aion/internal/types"
)

var version = "dev"

func main() {
	configPath := flag.String("config", "configs/aion.yaml", "path to AION config file")
	flag.Parse()

	slog.Info("starting AION", "version", version)

	// Load configuration.
	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	// Initialize telemetry store (SQLite).
	store, err := telemetry.NewStore(cfg.Telemetry.DBPath)
	if err != nil {
		slog.Error("failed to initialize telemetry store", "error", err)
		os.Exit(1)
	}
	defer store.Close()

	// Start async telemetry recorder.
	flushInterval, err := time.ParseDuration(cfg.Telemetry.FlushInterval)
	if err != nil {
		flushInterval = 5 * time.Second
	}
	recorder := telemetry.NewRecorder(store, cfg.Telemetry.BatchSize, flushInterval)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	recorder.Start(ctx)

	// Build pricing table.
	pricingTable := pricing.NewTable(cfg.Providers)

	// Build provider registry.
	healthMonitor := provider.NewHealthMonitor()
	registry := provider.NewRegistry()

	if cfg.Providers.OpenAI != nil {
		registry.Register(provider.NewOpenAI(cfg.Providers.OpenAI))
	}
	if cfg.Providers.Anthropic != nil {
		registry.Register(provider.NewAnthropic(cfg.Providers.Anthropic))
	}
	if cfg.Providers.OpenRouter != nil {
		registry.Register(provider.NewOpenRouter(cfg.Providers.OpenRouter))
	}
	if cfg.Providers.Bedrock != nil {
		registry.Register(provider.NewBedrock(cfg.Providers.Bedrock))
	}
	if cfg.Providers.Vertex != nil {
		registry.Register(provider.NewVertex(cfg.Providers.Vertex))
	}
	if cfg.Providers.Gemini != nil {
		registry.Register(provider.NewGemini(cfg.Providers.Gemini))
	}
	if cfg.Providers.Grok != nil {
		registry.Register(provider.NewGrok(cfg.Providers.Grok))
	}

	// Build classifier.
	cls := classifier.New(cfg.Routing.Classifier)

	// Build router.
	rtr := router.NewRouter(cfg, healthMonitor)

	// Build budget manager.
	budgetMgr := budget.NewManager(store)

	// Build proxy handler.
	proxyHandler := proxy.NewHandler(cls, rtr, registry, budgetMgr, pricingTable, recorder)

	// Build route handlers.
	handlers := server.RouteHandlers{
		ChatCompletion:    proxyHandler.ChatCompletion,
		AnthropicMessages: proxyHandler.AnthropicMessages,
		ListModels:        listModelsHandler(cfg),
		Health:            healthHandler(),
		MetricsSavings:    metricsHandler(store, "savings"),
		MetricsRouting:    metricsHandler(store, "routing"),
		MetricsCosts:      metricsHandler(store, "costs"),
	}

	// Build router (HTTP mux).
	mux := server.NewRouter(handlers)

	// Build middleware chain.
	var validator *apikey.Validator
	if cfg.Auth.Enabled {
		validator = apikey.NewValidator(cfg.Auth.Keys)
	}

	handler := server.Chain(
		mux,
		server.PanicRecoveryMiddleware,
		server.RequestIDMiddleware,
		server.LoggingMiddleware,
		server.CORSMiddleware,
		server.AuthMiddleware(validator),
	)

	// Parse timeouts.
	readTimeout, _ := time.ParseDuration(cfg.Server.ReadTimeout)
	if readTimeout == 0 {
		readTimeout = 30 * time.Second
	}
	writeTimeout, _ := time.ParseDuration(cfg.Server.WriteTimeout)
	if writeTimeout == 0 {
		writeTimeout = 60 * time.Second
	}
	shutdownTimeout, _ := time.ParseDuration(cfg.Server.ShutdownTimeout)
	if shutdownTimeout == 0 {
		shutdownTimeout = 10 * time.Second
	}

	// Create and start server.
	srv := server.New(cfg.Server.Port, handler, readTimeout, writeTimeout)

	// Graceful shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := srv.Start(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	slog.Info("AION is ready",
		"port", cfg.Server.Port,
		"providers", registry.List(),
		"auth", cfg.Auth.Enabled,
	)

	// Wait for shutdown signal.
	sig := <-sigCh
	slog.Info("received shutdown signal", "signal", sig)

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("server shutdown error", "error", err)
	}

	// Stop telemetry recorder (drains remaining events).
	cancel()
	recorder.Stop()

	slog.Info("AION shut down cleanly")
}

func healthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "ok",
			"version": version,
		})
	}
}

func listModelsHandler(cfg *config.Config) http.HandlerFunc {
	// Pre-build the model list from config.
	var models []types.ModelInfo
	now := time.Now().Unix()

	// Add AION virtual models.
	aionModels := []string{"aion-auto", "aion-escalate"}
	for _, id := range aionModels {
		models = append(models, types.ModelInfo{
			ID:      id,
			Object:  "model",
			Created: now,
			OwnedBy: "aion",
		})
	}

	// Add configured provider models.
	addModels := func(providerName string, pc *config.ProviderConfig) {
		if pc == nil {
			return
		}
		for _, m := range pc.Models {
			models = append(models, types.ModelInfo{
				ID:      m.ID,
				Object:  "model",
				Created: now,
				OwnedBy: providerName,
			})
		}
	}
	addModels("openai", cfg.Providers.OpenAI)
	addModels("anthropic", cfg.Providers.Anthropic)
	addModels("openrouter", cfg.Providers.OpenRouter)
	addModels("bedrock", cfg.Providers.Bedrock)
	addModels("vertex", cfg.Providers.Vertex)
	addModels("gemini", cfg.Providers.Gemini)
	addModels("grok", cfg.Providers.Grok)

	resp := types.ModelListResponse{
		Object: "list",
		Data:   models,
	}

	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func metricsHandler(store *telemetry.Store, metricType string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		from := r.URL.Query().Get("from")
		to := r.URL.Query().Get("to")

		if from == "" {
			from = time.Now().UTC().AddDate(0, -1, 0).Format("2006-01-02")
		}
		if to == "" {
			to = time.Now().UTC().AddDate(0, 0, 1).Format("2006-01-02")
		}

		w.Header().Set("Content-Type", "application/json")

		switch metricType {
		case "savings":
			report, err := store.QuerySavings(r.Context(), from, to)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			json.NewEncoder(w).Encode(report)

		case "routing":
			stats, err := store.QueryRoutingDistribution(r.Context(), from, to)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			json.NewEncoder(w).Encode(stats)

		case "costs":
			breakdown, err := store.QueryCostBreakdown(r.Context(), from, to)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			json.NewEncoder(w).Encode(breakdown)
		}
	}
}
