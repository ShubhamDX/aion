package router

import (
	"fmt"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

// Router selects the best model for a given complexity tier.
type Router struct {
	models   map[types.Tier][]ModelOption
	strategy Strategy
	health   HealthChecker
}

// NewRouter builds a router from config. It iterates all providers and their
// models, groups them by tier. Strategy is determined from config.Routing.Strategy.
// If health is nil, a default health checker that always returns true is used.
func NewRouter(cfg *config.Config, health HealthChecker) *Router {
	if health == nil {
		health = defaultHealthChecker{}
	}

	var strategy Strategy
	switch cfg.Routing.Strategy {
	case "fallback":
		strategy = FallbackStrategy{}
	default:
		strategy = CheapestStrategy{}
	}

	models := make(map[types.Tier][]ModelOption)

	// Helper to add models from a provider config.
	addProvider := func(providerName string, pc *config.ProviderConfig) {
		if pc == nil {
			return
		}
		for _, m := range pc.Models {
			opt := ModelOption{
				ID:               m.ID,
				Provider:         providerName,
				Tier:             types.Tier(m.Tier),
				InputPricePer1M:  m.InputPricePer1M,
				OutputPricePer1M: m.OutputPricePer1M,
			}
			tier := types.Tier(m.Tier)
			models[tier] = append(models[tier], opt)
		}
	}

	addProvider("openai", cfg.Providers.OpenAI)
	addProvider("anthropic", cfg.Providers.Anthropic)
	addProvider("openrouter", cfg.Providers.OpenRouter)
	addProvider("bedrock", cfg.Providers.Bedrock)
	addProvider("vertex", cfg.Providers.Vertex)
	addProvider("gemini", cfg.Providers.Gemini)
	addProvider("grok", cfg.Providers.Grok)

	// Register local provider models with $0 pricing.
	if lp := cfg.Providers.Local; lp != nil && lp.Enabled {
		for _, m := range lp.Models {
			opt := ModelOption{
				ID:               m.ID,
				Provider:         "local",
				Tier:             types.Tier(m.Tier),
				InputPricePer1M:  0,
				OutputPricePer1M: 0,
			}
			tier := types.Tier(m.Tier)
			models[tier] = append(models[tier], opt)
		}
	}

	return &Router{
		models:   models,
		strategy: strategy,
		health:   health,
	}
}

// Route selects the best model for the given tier.
func (r *Router) Route(tier types.Tier) (*ModelOption, error) {
	options, ok := r.models[tier]
	if !ok || len(options) == 0 {
		return nil, fmt.Errorf("router: no models configured for %s", tier)
	}
	return r.strategy.Select(options, r.health)
}

// RouteWithFallback tries the given tier first, then falls back to adjacent tiers.
// Tier 2 falls back to Tier 3, Tier 1 falls back to Tier 2.
func (r *Router) RouteWithFallback(tier types.Tier) (*ModelOption, error) {
	result, err := r.Route(tier)
	if err == nil {
		return result, nil
	}

	// Define fallback order: try the next higher tier.
	fallbacks := map[types.Tier]types.Tier{
		types.Tier1: types.Tier2,
		types.Tier2: types.Tier3,
	}

	next, hasFallback := fallbacks[tier]
	if !hasFallback {
		return nil, fmt.Errorf("router: no models available for %s and no fallback tier", tier)
	}

	result, err = r.Route(next)
	if err != nil {
		return nil, fmt.Errorf("router: no models available for %s or fallback %s", tier, next)
	}
	return result, nil
}

// FindModel looks up a specific model by ID across all tiers.
func (r *Router) FindModel(modelID string) (*ModelOption, error) {
	for _, options := range r.models {
		for i := range options {
			if options[i].ID == modelID {
				return &options[i], nil
			}
		}
	}
	return nil, fmt.Errorf("router: model %q not found", modelID)
}

// FindByProvider finds the cheapest healthy model from the named provider.
func (r *Router) FindByProvider(providerName string) (*ModelOption, error) {
	var candidates []ModelOption
	for _, options := range r.models {
		for _, opt := range options {
			if opt.Provider == providerName {
				candidates = append(candidates, opt)
			}
		}
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("router: no models found for provider %q", providerName)
	}
	return r.strategy.Select(candidates, r.health)
}
