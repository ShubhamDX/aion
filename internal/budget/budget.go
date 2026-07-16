package budget

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

// Reserve atomically adds a conservative request estimate when the projected
// usage remains within both limits. The returned date identifies the row that
// Settle must adjust after the provider finishes.
func (m *Manager) Reserve(ctx context.Context, apiKeyID string, estimate, dailyLimit, monthlyLimit float64) (string, error) {
	today := time.Now().UTC().Format("2006-01-02")
	yearMonth := today[:7]
	storageID := budgetStorageKey(apiKeyID)
	if err := m.store.MigrateBudgetUsageKey(ctx, apiKeyID, storageID); err != nil {
		return "", fmt.Errorf("budget: migrate legacy usage: %w", err)
	}
	dailyUsage, monthlyUsage, exceeded, err := m.store.ReserveBudgetUsage(
		ctx, storageID, today, yearMonth, estimate, dailyLimit, monthlyLimit,
	)
	if err != nil {
		return "", fmt.Errorf("budget: reserve usage: %w", err)
	}
	switch exceeded {
	case "daily":
		return "", fmt.Errorf("daily budget would be exceeded: $%.4f used + $%.4f reserved of $%.2f limit", dailyUsage, estimate, dailyLimit)
	case "monthly":
		return "", fmt.Errorf("monthly budget would be exceeded: $%.4f used + $%.4f reserved of $%.2f limit", monthlyUsage, estimate, monthlyLimit)
	default:
		return today, nil
	}
}

// Settle replaces the request estimate with known actual provider usage.
// Callers keep actual equal to the estimate when provider usage is unknown.
func (m *Manager) Settle(ctx context.Context, apiKeyID, date string, estimate, actual float64) error {
	if date == "" {
		return fmt.Errorf("budget: reservation date is required")
	}
	if estimate < 0 || actual < 0 {
		return fmt.Errorf("budget: estimate and actual cost must not be negative")
	}
	return m.store.AdjustBudgetUsage(ctx, budgetStorageKey(apiKeyID), date, actual-estimate)
}

func budgetStorageKey(apiKeyID string) string {
	digest := sha256.Sum256([]byte(apiKeyID))
	return "sha256:" + hex.EncodeToString(digest[:])
}
