package budget

import (
	"context"
	"fmt"
	"time"

	"github.com/ShubhamDX/aion/internal/telemetry"
)

// Manager enforces per-key spending limits.
type Manager struct {
	store *telemetry.Store
}

// NewManager creates a budget Manager backed by the given telemetry store.
func NewManager(store *telemetry.Store) *Manager {
	return &Manager{store: store}
}

// Check verifies if the API key can proceed given its budget limits.
// Returns nil if within budget, error describing which limit exceeded otherwise.
func (m *Manager) Check(ctx context.Context, apiKeyID string, dailyLimit, monthlyLimit float64) error {
	// Get today's date as "2006-01-02"
	today := time.Now().UTC().Format("2006-01-02")

	// Check daily limit
	if dailyLimit > 0 {
		usage, err := m.store.GetDailyUsage(ctx, apiKeyID, today)
		if err != nil {
			return fmt.Errorf("budget: check daily usage: %w", err)
		}
		if usage >= dailyLimit {
			return fmt.Errorf("daily budget exceeded: $%.4f used of $%.2f limit", usage, dailyLimit)
		}
	}

	// Check monthly limit
	if monthlyLimit > 0 {
		yearMonth := time.Now().UTC().Format("2006-01")
		usage, err := m.store.GetMonthlyUsage(ctx, apiKeyID, yearMonth)
		if err != nil {
			return fmt.Errorf("budget: check monthly usage: %w", err)
		}
		if usage >= monthlyLimit {
			return fmt.Errorf("monthly budget exceeded: $%.4f used of $%.2f limit", usage, monthlyLimit)
		}
	}

	return nil
}

// Record logs the actual cost after a request completes.
func (m *Manager) Record(ctx context.Context, apiKeyID string, costUSD float64) error {
	today := time.Now().UTC().Format("2006-01-02")
	return m.store.RecordBudgetUsage(ctx, apiKeyID, today, costUSD)
}
