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

// ExceededError is returned by Reserve when the projected cost would cross a
// configured daily or monthly limit. It carries the fields a caller needs to
// build a client-facing response (how much retry backoff to signal, when the
// window resets) instead of parsing the error string.
type ExceededError struct {
	// Scope is "daily" or "monthly".
	Scope string
	// Used is the amount already spent in the window before this request.
	Used float64
	// Estimate is the conservative reservation this request would have added.
	Estimate float64
	// Limit is the configured limit for Scope.
	Limit float64
	// ResetAt is when the window rolls over and spend starts counting from
	// zero again (next UTC midnight for daily, first of next UTC month for
	// monthly).
	ResetAt time.Time
}

func (e *ExceededError) Error() string {
	return fmt.Sprintf(
		"%s budget would be exceeded: $%.4f used + $%.4f reserved of $%.2f limit, resets %s",
		e.Scope, e.Used, e.Estimate, e.Limit, e.ResetAt.Format(time.RFC3339),
	)
}

// CustomerMessage renders the block in plain language for an end user: the
// limit and the reset time, without the internal "used + reserved" bookkeeping
// in Error(). Use this for any response the calling client or its user will
// see; keep Error() for logs.
func (e *ExceededError) CustomerMessage() string {
	return fmt.Sprintf(
		"%s usage limit reached. Resets %s.",
		e.Scope, e.ResetAt.Format("2006-01-02 15:04 MST"),
	)
}

// Reserve atomically adds a conservative request estimate when the projected
// usage remains within both limits. The returned date identifies the row that
// Settle must adjust after the provider finishes. A non-nil error is always
// either an *ExceededError (the request is over budget) or a wrapped storage
// error (the reservation could not be attempted).
func (m *Manager) Reserve(ctx context.Context, apiKeyID string, estimate, dailyLimit, monthlyLimit float64) (string, error) {
	now := time.Now().UTC()
	today := now.Format("2006-01-02")
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
		return "", &ExceededError{Scope: "daily", Used: dailyUsage, Estimate: estimate, Limit: dailyLimit, ResetAt: nextUTCMidnight(now)}
	case "monthly":
		return "", &ExceededError{Scope: "monthly", Used: monthlyUsage, Estimate: estimate, Limit: monthlyLimit, ResetAt: nextUTCMonth(now)}
	default:
		return today, nil
	}
}

// nextUTCMidnight returns the start of the next UTC day after now, matching
// the daily window ReserveBudgetUsage keys usage rows by (YYYY-MM-DD).
func nextUTCMidnight(now time.Time) time.Time {
	y, mo, d := now.Date()
	return time.Date(y, mo, d, 0, 0, 0, 0, time.UTC).AddDate(0, 0, 1)
}

// nextUTCMonth returns the start of the next UTC calendar month after now,
// matching the monthly window ReserveBudgetUsage keys usage rows by
// (YYYY-MM) prefix.
func nextUTCMonth(now time.Time) time.Time {
	y, mo, _ := now.Date()
	return time.Date(y, mo, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 1, 0)
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
