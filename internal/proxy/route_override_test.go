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
