package config

import (
	"errors"
	"fmt"
)

// applyDefaults fills in zero-valued fields with sensible production defaults.
func applyDefaults(cfg *Config) {
	// Server defaults.
	if cfg.Server.Port == 0 {
		cfg.Server.Port = 8080
	}
	if cfg.Server.ReadTimeout == "" {
		cfg.Server.ReadTimeout = "30s"
	}
	if cfg.Server.WriteTimeout == "" {
		cfg.Server.WriteTimeout = "60s"
	}
	if cfg.Server.ShutdownTimeout == "" {
		cfg.Server.ShutdownTimeout = "10s"
	}

	// Routing defaults.
	if cfg.Routing.Strategy == "" {
		cfg.Routing.Strategy = "cheapest"
	}
	if cfg.Routing.Classifier.Tier1Threshold == 0 {
		cfg.Routing.Classifier.Tier1Threshold = 0.35
	}
	if cfg.Routing.Classifier.Tier2Threshold == 0 {
		cfg.Routing.Classifier.Tier2Threshold = 0.70
	}

	// Telemetry defaults.
	if cfg.Telemetry.DBPath == "" {
		cfg.Telemetry.DBPath = "./data/aion.db"
	}
	if cfg.Telemetry.BatchSize == 0 {
		cfg.Telemetry.BatchSize = 100
	}
	if cfg.Telemetry.FlushInterval == "" {
		cfg.Telemetry.FlushInterval = "5s"
	}
}

// validate checks the configuration for logical errors.
func validate(cfg *Config) error {
	var errs []error

	if cfg.Server.Port <= 0 {
		errs = append(errs, fmt.Errorf("server.port must be > 0, got %d", cfg.Server.Port))
	}

	// At least one provider with at least one model must be configured.
	if !hasProvider(cfg) {
		errs = append(errs, errors.New("at least one provider with at least one model must be configured"))
	}

	// Classifier thresholds: 0 < tier1 < tier2 < 1.
	t1 := cfg.Routing.Classifier.Tier1Threshold
	t2 := cfg.Routing.Classifier.Tier2Threshold
	if t1 <= 0 || t1 >= t2 || t2 >= 1 {
		errs = append(errs, fmt.Errorf(
			"classifier thresholds must satisfy 0 < tier1 < tier2 < 1, got tier1=%g tier2=%g", t1, t2,
		))
	}

	// If auth is enabled, at least one key is required.
	if cfg.Auth.Enabled && len(cfg.Auth.Keys) == 0 {
		errs = append(errs, errors.New("auth is enabled but no keys are configured"))
	}

	return errors.Join(errs...)
}

// hasProvider returns true if at least one provider is configured with at
// least one model.
func hasProvider(cfg *Config) bool {
	providers := []*ProviderConfig{
		cfg.Providers.OpenAI,
		cfg.Providers.Anthropic,
		cfg.Providers.OpenRouter,
	}
	for _, p := range providers {
		if p != nil && len(p.Models) > 0 {
			return true
		}
	}
	return false
}
