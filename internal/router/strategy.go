package router

import (
	"errors"
	"sort"

	"github.com/ShubhamDX/aion/internal/types"
)

// ErrNoHealthyModel is returned when no healthy model is available for routing.
var ErrNoHealthyModel = errors.New("router: no healthy model available")

// ModelOption represents a model that can be selected for routing.
type ModelOption struct {
	ID               string
	Provider         string
	Tier             types.Tier
	InputPricePer1M  float64
	OutputPricePer1M float64
	MaxTokens        int
}

// CombinedPrice returns total price per 1M tokens (input+output).
func (m *ModelOption) CombinedPrice() float64 {
	return m.InputPricePer1M + m.OutputPricePer1M
}

// HealthChecker checks if a model/provider is healthy.
type HealthChecker interface {
	IsHealthy(provider, model string) bool
}

// defaultHealthChecker always reports models as healthy.
type defaultHealthChecker struct{}

func (defaultHealthChecker) IsHealthy(string, string) bool { return true }

// Strategy selects a model from available options.
type Strategy interface {
	Select(options []ModelOption, health HealthChecker) (*ModelOption, error)
}

// CheapestStrategy picks the cheapest healthy model.
type CheapestStrategy struct{}

// Select sorts options by CombinedPrice ascending and returns the first healthy one.
func (CheapestStrategy) Select(options []ModelOption, health HealthChecker) (*ModelOption, error) {
	if len(options) == 0 {
		return nil, ErrNoHealthyModel
	}

	sorted := make([]ModelOption, len(options))
	copy(sorted, options)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].CombinedPrice() < sorted[j].CombinedPrice()
	})

	for i := range sorted {
		if health.IsHealthy(sorted[i].Provider, sorted[i].ID) {
			return &sorted[i], nil
		}
	}
	return nil, ErrNoHealthyModel
}

// FallbackStrategy tries models in order (sorted by price ascending), skipping unhealthy ones.
type FallbackStrategy struct{}

// Select iterates models sorted by price and returns the first healthy one.
func (FallbackStrategy) Select(options []ModelOption, health HealthChecker) (*ModelOption, error) {
	if len(options) == 0 {
		return nil, ErrNoHealthyModel
	}

	sorted := make([]ModelOption, len(options))
	copy(sorted, options)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].CombinedPrice() < sorted[j].CombinedPrice()
	})

	for i := range sorted {
		if health.IsHealthy(sorted[i].Provider, sorted[i].ID) {
			return &sorted[i], nil
		}
	}
	return nil, ErrNoHealthyModel
}
