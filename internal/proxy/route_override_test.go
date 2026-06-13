package proxy

import (
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/types"
)

// twoTierHandler builds a Handler whose router has a cheap tier-1 pair and a
// pricey tier-2 model, so cheapest-selection within a tier is observable.
func twoTierHandler(health router.HealthChecker) *Handler {
	cfg := &config.Config{}
	cfg.Providers.Bedrock = &config.ProviderConfig{
		Models: []config.ModelConfig{
			{ID: "cheap-haiku", Tier: 1, InputPricePer1M: 0.8, OutputPricePer1M: 4},
			{ID: "pricey-haiku", Tier: 1, InputPricePer1M: 2, OutputPricePer1M: 10},
			{ID: "sonnet", Tier: 2, InputPricePer1M: 3, OutputPricePer1M: 15},
		},
	}
	return &Handler{router: router.NewRouter(cfg, health)}
}

// mixedProviderHandler has a CHEAPER tier-1 model on a non-bedrock provider, so
// a provider allowlist must measurably change which model a tier route picks.
func mixedProviderHandler(health router.HealthChecker) *Handler {
	cfg := &config.Config{}
	cfg.Providers.Bedrock = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "bedrock-haiku", Tier: 1, InputPricePer1M: 0.8, OutputPricePer1M: 4}},
	}
	cfg.Providers.Local = &config.LocalProviderConfig{Enabled: true, Models: []config.ModelConfig{
		{ID: "local-tiny", Tier: 1, InputPricePer1M: 0, OutputPricePer1M: 0}, // cheaper, but disallowed provider
	}}
	cfg.Providers.Anthropic = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "sonnet", Tier: 2, InputPricePer1M: 3, OutputPricePer1M: 15}},
	}
	return &Handler{router: router.NewRouter(cfg, health)}
}

// downModel returns a health checker that reports one model id as unhealthy.
type downModel string

func (d downModel) IsHealthy(_, model string) bool { return model != string(d) }

func TestResolveRouteOverride(t *testing.T) {
	h := twoTierHandler(nil)
	sonnet, err := h.router.FindModel("sonnet")
	if err != nil {
		t.Fatalf("seed model: %v", err)
	}

	t.Run("tier override picks cheapest in tier", func(t *testing.T) {
		m, changed, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedTierOverride: 1}, sonnet)
		if err != nil || !changed {
			t.Fatalf("changed=%v err=%v", changed, err)
		}
		if m.ID != "cheap-haiku" {
			t.Fatalf("got %q want cheap-haiku", m.ID)
		}
	})

	t.Run("explicit model wins over tier", func(t *testing.T) {
		m, changed, err := h.resolveRouteOverride(types.PreRequestDecision{
			RoutedModelOverride: "pricey-haiku", RoutedTierOverride: 1,
		}, sonnet)
		if err != nil || !changed || m.ID != "pricey-haiku" {
			t.Fatalf("got %q changed=%v err=%v", m.ID, changed, err)
		}
	})

	t.Run("unknown model fails closed", func(t *testing.T) {
		if _, _, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedModelOverride: "nope"}, sonnet); err == nil {
			t.Fatal("want error for unknown model")
		}
	})

	t.Run("empty tier with no models fails closed", func(t *testing.T) {
		if _, _, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedTierOverride: 3}, sonnet); err == nil {
			t.Fatal("want error: no tier-3 model")
		}
	})

	t.Run("strict tier: empty tier does NOT fall back to an adjacent tier", func(t *testing.T) {
		// tier-3 is empty, tier-2 (sonnet) exists. A fallback router would land
		// on tier-2; strict must fail closed instead.
		_, _, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedTierOverride: 3}, sonnet)
		if err == nil {
			t.Fatal("tier override must not widen to a populated adjacent tier")
		}
	})

	t.Run("no target is a no-op", func(t *testing.T) {
		m, changed, err := h.resolveRouteOverride(types.PreRequestDecision{}, sonnet)
		if err != nil || changed || m.ID != "sonnet" {
			t.Fatalf("got %q changed=%v err=%v", m.ID, changed, err)
		}
	})

	t.Run("override to same model is not a change", func(t *testing.T) {
		_, changed, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedModelOverride: "sonnet"}, sonnet)
		if err != nil || changed {
			t.Fatalf("changed=%v err=%v", changed, err)
		}
	})
}

func TestResolveRouteOverrideHealthAware(t *testing.T) {
	// cheapest tier-1 model is down -> tier override must skip it for the next
	// cheapest healthy one, NOT fail.
	h := twoTierHandler(downModel("cheap-haiku"))
	sonnet, _ := h.router.FindModel("sonnet")
	m, changed, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedTierOverride: 1}, sonnet)
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
	if m.ID != "pricey-haiku" {
		t.Fatalf("got %q want pricey-haiku (cheapest healthy)", m.ID)
	}
}

// allowSet builds a health checker that reports the named models unhealthy.
type downSet map[string]bool

func (d downSet) IsHealthy(_, model string) bool { return !d[model] }

func TestResolveRouteOverrideStrictTierFailsClosed(t *testing.T) {
	// Finding 1: tier-1 has models but ALL are unhealthy; tier-2 is healthy. A
	// fallback router would silently route to tier-2. Strict must fail closed.
	h := twoTierHandler(downSet{"cheap-haiku": true, "pricey-haiku": true})
	sonnet, _ := h.router.FindModel("sonnet")
	_, _, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedTierOverride: 1}, sonnet)
	if err == nil {
		t.Fatal("tier-1 all unhealthy must fail closed, not fall back to tier-2")
	}
}

func TestResolveRouteOverrideProviderConstrained(t *testing.T) {
	// Finding 2: a cheaper tier-1 model exists on a disallowed provider (local).
	// A tier route constrained to bedrock must pick the bedrock model, never the
	// cheaper local one.
	h := mixedProviderHandler(nil)
	sonnet, _ := h.router.FindModel("sonnet")

	t.Run("allowlist excludes cheaper disallowed provider", func(t *testing.T) {
		m, changed, err := h.resolveRouteOverride(types.PreRequestDecision{
			RoutedTierOverride: 1, RoutedAllowedProviders: []string{"bedrock"},
		}, sonnet)
		if err != nil || !changed {
			t.Fatalf("changed=%v err=%v", changed, err)
		}
		if m.ID != "bedrock-haiku" {
			t.Fatalf("got %q want bedrock-haiku (local-tiny is cheaper but disallowed)", m.ID)
		}
	})

	t.Run("no allowlist picks the global cheapest", func(t *testing.T) {
		m, _, err := h.resolveRouteOverride(types.PreRequestDecision{RoutedTierOverride: 1}, sonnet)
		if err != nil || m.ID != "local-tiny" {
			t.Fatalf("got %q err=%v want local-tiny (cheapest, unconstrained)", m.ID, err)
		}
	})

	t.Run("allowlist with no model in tier fails closed", func(t *testing.T) {
		_, _, err := h.resolveRouteOverride(types.PreRequestDecision{
			RoutedTierOverride: 1, RoutedAllowedProviders: []string{"vertex"},
		}, sonnet)
		if err == nil {
			t.Fatal("no tier-1 vertex model: must fail closed")
		}
	})
}
