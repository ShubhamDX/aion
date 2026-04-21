package router

import (
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
)

func TestRouter_LocalProviderRegistered(t *testing.T) {
	cfg := &config.Config{
		Routing: config.RoutingConfig{Strategy: "cheapest"},
		Providers: config.ProvidersConfig{
			Local: &config.LocalProviderConfig{
				Enabled: true,
				Models: []config.ModelConfig{
					{ID: "qwen2.5-1.5b-instruct", Tier: 1},
				},
			},
			OpenAI: &config.ProviderConfig{
				Models: []config.ModelConfig{
					{ID: "gpt-4o-mini", Tier: 1, InputPricePer1M: 0.15, OutputPricePer1M: 0.60},
				},
			},
		},
	}

	r := NewRouter(cfg, nil)

	// Cheapest in tier 1 should be the local model ($0 beats gpt-4o-mini).
	opt, err := r.Route(1)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if opt.Provider != "local" {
		t.Fatalf("want provider=local, got %q (model=%s)", opt.Provider, opt.ID)
	}
	if opt.CombinedPrice() != 0 {
		t.Fatalf("want $0 combined price, got %v", opt.CombinedPrice())
	}
}

func TestRouter_LocalDisabled(t *testing.T) {
	cfg := &config.Config{
		Routing: config.RoutingConfig{Strategy: "cheapest"},
		Providers: config.ProvidersConfig{
			Local: &config.LocalProviderConfig{
				Enabled: false, // disabled — should not register
				Models: []config.ModelConfig{
					{ID: "qwen2.5-1.5b-instruct", Tier: 1},
				},
			},
			OpenAI: &config.ProviderConfig{
				Models: []config.ModelConfig{
					{ID: "gpt-4o-mini", Tier: 1, InputPricePer1M: 0.15, OutputPricePer1M: 0.60},
				},
			},
		},
	}

	r := NewRouter(cfg, nil)

	if _, err := r.FindByProvider("local"); err == nil {
		t.Fatalf("expected FindByProvider(local) to error when local is disabled")
	}

	opt, err := r.Route(1)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if opt.Provider != "openai" {
		t.Fatalf("want provider=openai, got %q", opt.Provider)
	}
}

func TestRouter_FindByProvider(t *testing.T) {
	cfg := &config.Config{
		Routing: config.RoutingConfig{Strategy: "cheapest"},
		Providers: config.ProvidersConfig{
			Anthropic: &config.ProviderConfig{
				Models: []config.ModelConfig{
					{ID: "claude-haiku-3-5", Tier: 1, InputPricePer1M: 0.80, OutputPricePer1M: 4.00},
					{ID: "claude-sonnet-4", Tier: 2, InputPricePer1M: 3.00, OutputPricePer1M: 15.00},
				},
			},
			OpenAI: &config.ProviderConfig{
				Models: []config.ModelConfig{
					{ID: "gpt-4o-mini", Tier: 1, InputPricePer1M: 0.15, OutputPricePer1M: 0.60},
				},
			},
		},
	}

	r := NewRouter(cfg, nil)

	opt, err := r.FindByProvider("anthropic")
	if err != nil {
		t.Fatalf("FindByProvider(anthropic): %v", err)
	}
	if opt.Provider != "anthropic" {
		t.Fatalf("want provider=anthropic, got %q", opt.Provider)
	}
	// Cheapest among anthropic models -> haiku.
	if opt.ID != "claude-haiku-3-5" {
		t.Fatalf("want haiku, got %q", opt.ID)
	}

	if _, err := r.FindByProvider("bedrock"); err == nil {
		t.Fatalf("expected error for unregistered provider")
	}
}
