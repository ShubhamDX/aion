package types

import (
	"fmt"
	"time"
)

// Tier represents the complexity tier for routing decisions.
type Tier int

const (
	Tier1 Tier = 1 // Simple tasks - cheapest models
	Tier2 Tier = 2 // Moderate tasks - mid-range models
	Tier3 Tier = 3 // Complex tasks - most capable models
)

// String returns a human-readable representation of the tier.
func (t Tier) String() string {
	switch t {
	case Tier1:
		return "Tier1"
	case Tier2:
		return "Tier2"
	case Tier3:
		return "Tier3"
	default:
		return fmt.Sprintf("Tier(%d)", int(t))
	}
}

// AIONPreferences allows callers to provide routing hints in the request.
type AIONPreferences struct {
	PreferredTier *int     `json:"preferred_tier,omitempty"`
	MaxCostUSD    *float64 `json:"max_cost_usd,omitempty"`
	PreferQuality bool     `json:"prefer_quality,omitempty"`
	PreferSpeed   bool     `json:"prefer_speed,omitempty"`
}

// RoutingDecision captures the outcome of the classification and routing process.
type RoutingDecision struct {
	Tier     Tier               `json:"tier"`
	Model    string             `json:"model"`
	Provider string             `json:"provider"`
	Score    float64            `json:"score"`
	Signals  map[string]float64 `json:"signals,omitempty"`
}

// RequestEvent records telemetry for a single proxied request.
type RequestEvent struct {
	RequestID  string    `json:"request_id"`
	APIKeyID   string    `json:"api_key_id"`
	Tier       int       `json:"tier"`
	Model      string    `json:"model"`
	Provider   string    `json:"provider"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	CostUSD    float64   `json:"cost_usd"`
	SavingsUSD float64   `json:"savings_usd"`
	LatencyMS  int64     `json:"latency_ms"`
	StatusCode int       `json:"status_code"`
	Stream     bool      `json:"stream"`
	CreatedAt  time.Time `json:"created_at"`
}
