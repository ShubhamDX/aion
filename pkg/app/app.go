package app

import (
	"context"
	"encoding/json"
	"fmt"
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
	pkgclassifier "github.com/ShubhamDX/aion/pkg/classifier"
	pkgtypes "github.com/ShubhamDX/aion/pkg/types"
)

// Version is the application version string. Set via ldflags at build time.
var Version = "dev"

// Options configures the application builder.
type Options struct {
	// Classifier overrides the default TF-IDF classifier. If nil, the built-in
	// OSS classifier is used.
	Classifier pkgclassifier.Classifier

	// ConfigPath is the path to the YAML configuration file.
	ConfigPath string

	// Config supplies an already loaded configuration. An embedding product can
	// use this to apply its own configuration layering once and pass the exact
	// result to the OSS runtime. When set, ConfigPath is ignored.
	Config *config.Config

	// GatewayHooks is the optional pre-request / post-response extension surface
	// (an embedding product wraps the proxy request lifecycle here). nil leaves
	// OSS behavior unchanged.
	GatewayHooks *pkgtypes.GatewayHooks

	// ExtraRoutes, if set, is called with the gateway mux after the built-in
	// routes are registered and before the middleware chain wraps it, so an
	// embedding product can mount additional handlers (e.g. a license-status
	// probe or an approval-queue API). Routes under /healthz/ bypass auth.
	// ExtraAuthBypassPrefixes can opt extra mounted route prefixes out of API-key
	// auth so the embedding product can install its own auth. nil leaves OSS
	// behavior unchanged.
	ExtraRoutes func(mux *http.ServeMux)

	// ExtraAuthBypassPrefixes are route prefixes mounted by ExtraRoutes that
	// should bypass OSS API-key auth. Use only when the embedding product wraps
	// that prefix in its own auth layer. Empty by default.
	ExtraAuthBypassPrefixes []string
}

// App encapsulates the full AION gateway runtime.
type App struct {
	cfg           *config.Config
	srv           *server.Server
	recorder      *telemetry.Recorder
	store         *telemetry.Store
	cancel        context.CancelFunc
	localProvider *provider.LocalProvider
	proxyHandler  *proxy.Handler
}

// Build constructs an App from the given options. It loads config, initializes
// telemetry, providers, router, and HTTP server. If opts.Classifier is nil the
// built-in TF-IDF classifier is used.
func Build(opts Options) (*App, error) {
	cfg, err := loadConfig(opts)
	if err != nil {
		return nil, err
	}

	// Initialize telemetry store (SQLite).
	store, err := telemetry.NewStore(cfg.Telemetry.DBPath)
	if err != nil {
		return nil, err
	}

	// Start async telemetry recorder.
	flushInterval, err := time.ParseDuration(cfg.Telemetry.FlushInterval)
	if err != nil {
		flushInterval = 5 * time.Second
	}
	recorder := telemetry.NewRecorder(store, cfg.Telemetry.BatchSize, flushInterval)
	ctx, cancel := context.WithCancel(context.Background())
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

	// Local provider (llama.cpp).
	var localProvider *provider.LocalProvider
	if lp := cfg.Providers.Local; lp != nil && lp.Enabled {
		localProvider = provider.NewLocal(lp)

		// Managed mode: start llama-server subprocess.
		if lp.Managed != nil {
			proc := provider.NewLlamaProcess(lp.Managed)
			if err := proc.Start(); err != nil {
				cancel()
				store.Close()
				return nil, fmt.Errorf("local provider: %w", err)
			}
			localProvider.SetProcess(proc)

			// Wait for the server to become ready.
			readyTimeout, _ := time.ParseDuration(lp.Managed.ReadyTimeout)
			if readyTimeout == 0 {
				readyTimeout = 120 * time.Second
			}
			if err := localProvider.WaitReady(ctx, readyTimeout); err != nil {
				_ = proc.Stop()
				cancel()
				store.Close()
				return nil, fmt.Errorf("local provider: %w", err)
			}
		}

		registry.Register(localProvider)
	}

	// Resolve classifier.
	var cls pkgclassifier.Classifier
	if opts.Classifier != nil {
		cls = opts.Classifier
	} else {
		cls = classifier.New(cfg.Routing.Classifier)
	}

	// Build router.
	rtr := router.NewRouter(cfg, healthMonitor)

	// Build budget manager.
	budgetMgr := budget.NewManager(store)

	// Build proxy handler.
	proxyHandler := proxy.NewHandler(cls, rtr, registry, budgetMgr, pricingTable, recorder)
	if opts.GatewayHooks != nil {
		proxyHandler.SetGatewayHooks(opts.GatewayHooks)
	}

	// Build route handlers.
	handlers := server.RouteHandlers{
		ChatCompletion:    proxyHandler.ChatCompletion,
		Responses:         proxyHandler.Responses,
		AnthropicMessages: proxyHandler.AnthropicMessages,
		ListModels:        listModelsHandler(cfg),
		Health:            healthHandler(),
		MetricsSavings:    metricsHandler(store, "savings"),
		MetricsRouting:    metricsHandler(store, "routing"),
		MetricsCosts:      metricsHandler(store, "costs"),
	}

	// Build HTTP mux.
	mux := server.NewRouter(handlers)

	// Let an embedding product mount extra routes (license status, approval API)
	// before the middleware chain wraps the mux.
	if opts.ExtraRoutes != nil {
		opts.ExtraRoutes(mux)
	}

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
		server.AuthMiddlewareWithOptions(validator, server.AuthOptions{
			ExtraBypassPrefixes: opts.ExtraAuthBypassPrefixes,
		}),
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

	srv := server.New(cfg.Server.Port, handler, readTimeout, writeTimeout)

	slog.Info("AION built",
		"port", cfg.Server.Port,
		"providers", registry.List(),
		"auth", cfg.Auth.Enabled,
	)

	return &App{
		cfg:           cfg,
		srv:           srv,
		recorder:      recorder,
		store:         store,
		cancel:        cancel,
		localProvider: localProvider,
		proxyHandler:  proxyHandler,
	}, nil
}

func loadConfig(opts Options) (*config.Config, error) {
	if opts.Config != nil {
		if err := config.Prepare(opts.Config); err != nil {
			return nil, fmt.Errorf("config: validation: %w", err)
		}
		return opts.Config, nil
	}
	return config.Load(opts.ConfigPath)
}

// DrainPostResponse blocks until all in-flight asynchronous PostResponse hooks
// finish. An embedding product calls this during shutdown, after the HTTP
// server has stopped and before it closes its evidence store, so an async
// schema/evidence hook is never cut off mid-write.
func (a *App) DrainPostResponse() { a.proxyHandler.DrainPostResponse() }

// Config returns the loaded configuration.
func (a *App) Config() *config.Config {
	return a.cfg
}

// Handler returns the composed HTTP handler, so an embedder or test can drive
// the gateway via httptest without binding a port (used for end-to-end tests).
func (a *App) Handler() http.Handler {
	return a.srv.Handler()
}

// Run starts the HTTP server and blocks until a SIGINT or SIGTERM is received,
// then performs graceful shutdown.
func (a *App) Run() error {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := a.srv.Start(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	slog.Info("AION is ready",
		"port", a.cfg.Server.Port,
		"version", Version,
	)

	sig := <-sigCh
	slog.Info("received shutdown signal", "signal", sig)

	shutdownTimeout, _ := time.ParseDuration(a.cfg.Server.ShutdownTimeout)
	if shutdownTimeout == 0 {
		shutdownTimeout = 10 * time.Second
	}

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer shutdownCancel()

	if err := a.srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("server shutdown error", "error", err)
	}

	// Drain in-flight async PostResponse hooks before tearing down: the server no
	// longer accepts requests, so this finishes the evidence/schema writes already
	// dispatched, ahead of any store close (the embedding product also closes its
	// own store via its runtime closer).
	a.proxyHandler.DrainPostResponse()

	// Shutdown local provider (managed llama-server).
	if a.localProvider != nil {
		if err := a.localProvider.Shutdown(); err != nil {
			slog.Error("local provider shutdown error", "error", err)
		}
	}

	// Stop telemetry recorder (drains remaining events).
	a.cancel()
	a.recorder.Stop()
	a.store.Close()

	slog.Info("AION shut down cleanly")
	return nil
}

func healthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "ok",
			"version": Version,
		})
	}
}

func listModelsHandler(cfg *config.Config) http.HandlerFunc {
	var models []types.ModelInfo
	now := time.Now().Unix()

	aionModels := []string{"aion-auto", "aion-escalate", "aion-local"}
	for _, id := range aionModels {
		models = append(models, types.ModelInfo{
			ID:      id,
			Object:  "model",
			Created: now,
			OwnedBy: "aion",
		})
	}

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

	// Add local provider models.
	if lp := cfg.Providers.Local; lp != nil && lp.Enabled {
		for _, m := range lp.Models {
			models = append(models, types.ModelInfo{
				ID:      m.ID,
				Object:  "model",
				Created: now,
				OwnedBy: "local",
			})
		}
	}

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
