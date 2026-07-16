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

	// Local provider defaults.
	applyLocalDefaults(cfg)
}

func applyLocalDefaults(cfg *Config) {
	lp := cfg.Providers.Local
	if lp == nil || !lp.Enabled {
		return
	}

	if lp.BaseURL == "" {
		lp.BaseURL = "http://localhost:8081/v1"
	}

	if lp.Managed != nil {
		m := lp.Managed
		if m.Port == 0 {
			m.Port = 8081
		}
		if m.ContextSize == 0 {
			m.ContextSize = 4096
		}
		if m.BatchSize == 0 {
			m.BatchSize = 512
		}
		if m.ReadyTimeout == "" {
			m.ReadyTimeout = "120s"
		}
	}

	// Force $0 pricing for all local models.
	for i := range lp.Models {
		lp.Models[i].InputPricePer1M = 0
		lp.Models[i].OutputPricePer1M = 0
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

	// Local provider validation.
	if lp := cfg.Providers.Local; lp != nil && lp.Enabled {
		if len(lp.Models) == 0 {
			errs = append(errs, errors.New("local provider is enabled but no models are configured"))
		}
		if lp.Managed != nil && lp.Managed.ModelPath == "" {
			errs = append(errs, errors.New("local.managed.model_path is required in managed mode"))
		}
	}

	// If auth is enabled, at least one key is required.
	if cfg.Auth.Enabled && len(cfg.Auth.Keys) == 0 && cfg.Auth.ManagedKeysPath == "" {
		errs = append(errs, errors.New("auth is enabled but no keys are configured"))
	}

	if bedrock := cfg.Providers.Bedrock; bedrock != nil {
		switch bedrock.CredentialMode {
		case "", "bearer", "aws_sdk", "assume_role":
		default:
			errs = append(errs, fmt.Errorf("bedrock.credential_mode must be bearer, aws_sdk or assume_role, got %q", bedrock.CredentialMode))
		}
		if bedrock.CredentialMode == "bearer" && bedrock.APIKey == "" {
			errs = append(errs, errors.New("bedrock.api_key is required for bearer credential mode"))
		}
		if bedrock.CredentialMode == "assume_role" && bedrock.RoleARN == "" {
			errs = append(errs, errors.New("bedrock.role_arn is required for assume_role credential mode"))
		}
	}

	for providerName, models := range configuredModels(cfg) {
		for _, model := range models {
			if model.Tier < 1 || model.Tier > 3 {
				errs = append(errs, fmt.Errorf("%s model %q tier must be 1, 2 or 3, got %d", providerName, model.ID, model.Tier))
			}
		}
	}

	return errors.Join(errs...)
}

func configuredModels(cfg *Config) map[string][]ModelConfig {
	models := map[string][]ModelConfig{}
	providers := map[string]*ProviderConfig{
		"openai": cfg.Providers.OpenAI, "anthropic": cfg.Providers.Anthropic,
		"openrouter": cfg.Providers.OpenRouter, "bedrock": cfg.Providers.Bedrock,
		"vertex": cfg.Providers.Vertex, "gemini": cfg.Providers.Gemini,
		"grok": cfg.Providers.Grok,
	}
	for name, provider := range providers {
		if provider != nil {
			models[name] = provider.Models
		}
	}
	if local := cfg.Providers.Local; local != nil && local.Enabled {
		models["local"] = local.Models
	}
	return models
}

// hasProvider returns true if at least one provider is configured with at
// least one model.
func hasProvider(cfg *Config) bool {
	providers := []*ProviderConfig{
		cfg.Providers.OpenAI,
		cfg.Providers.Anthropic,
		cfg.Providers.OpenRouter,
		cfg.Providers.Bedrock,
		cfg.Providers.Vertex,
		cfg.Providers.Gemini,
		cfg.Providers.Grok,
	}
	for _, p := range providers {
		if p != nil && len(p.Models) > 0 {
			return true
		}
	}
	if lp := cfg.Providers.Local; lp != nil && lp.Enabled && len(lp.Models) > 0 {
		return true
	}
	return false
}
