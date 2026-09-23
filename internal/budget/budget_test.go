package budget

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ShubhamDX/aion/internal/telemetry"
)

func newTestManager(t *testing.T) (*Manager, *telemetry.Store) {
	t.Helper()
	store, err := telemetry.NewStore(filepath.Join(t.TempDir(), "telemetry.db"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return NewManager(store), store
}

func TestReserveRejectsProjectedOverspendAndSettleReleasesRemainder(t *testing.T) {
	manager, store := newTestManager(t)
	ctx := context.Background()
	date, err := manager.Reserve(ctx, "tester", 4.50, 5, 20)
	if err != nil {
		t.Fatalf("Reserve first request: %v", err)
	}
	if _, err := manager.Reserve(ctx, "tester", 0.51, 5, 20); err == nil {
		t.Fatal("Reserve must reject a request whose estimate crosses the daily limit")
	}
	if err := manager.Settle(ctx, "tester", date, 4.50, 1.00); err != nil {
		t.Fatalf("Settle: %v", err)
	}
	if _, err := manager.Reserve(ctx, "tester", 4.00, 5, 20); err != nil {
		t.Fatalf("Reserve after settlement: %v", err)
	}

	usage, err := store.GetDailyUsage(ctx, budgetStorageKey("tester"), time.Now().UTC().Format("2006-01-02"))
	if err != nil {
		t.Fatalf("GetDailyUsage: %v", err)
	}
	if usage != 5 {
		t.Fatalf("daily usage = %.2f, want 5.00", usage)
	}
}

func TestReserveMigratesLegacyRawKeyRows(t *testing.T) {
	manager, store := newTestManager(t)
	ctx := context.Background()
	today := time.Now().UTC().Format("2006-01-02")
	if err := store.RecordBudgetUsage(ctx, "secret-value", today, 1.25); err != nil {
		t.Fatalf("RecordBudgetUsage: %v", err)
	}
	if _, err := manager.Reserve(ctx, "secret-value", 0.75, 2, 10); err != nil {
		t.Fatalf("Reserve: %v", err)
	}
	if legacy, err := store.GetDailyUsage(ctx, "secret-value", today); err != nil || legacy != 0 {
		t.Fatalf("legacy usage = %.2f, err=%v, want 0", legacy, err)
	}
	if migrated, err := store.GetDailyUsage(ctx, budgetStorageKey("secret-value"), today); err != nil || migrated != 2 {
		t.Fatalf("migrated usage = %.2f, err=%v, want 2", migrated, err)
	}
}

func TestReserveSerializesConcurrentRequests(t *testing.T) {
	manager, _ := newTestManager(t)
	ctx := context.Background()
	start := make(chan struct{})
	results := make(chan error, 2)
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := manager.Reserve(ctx, "tester", 3, 5, 20)
			results <- err
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	succeeded := 0
	failed := 0
	for err := range results {
		if err == nil {
			succeeded++
		} else {
			failed++
		}
	}
	if succeeded != 1 || failed != 1 {
		t.Fatalf("concurrent reservations: succeeded=%d failed=%d, want 1 and 1", succeeded, failed)
	}
}

func TestReserveEnforcesMonthlyLimit(t *testing.T) {
	manager, _ := newTestManager(t)
	ctx := context.Background()
	date, err := manager.Reserve(ctx, "tester", 2, 10, 3)
	if err != nil {
		t.Fatalf("Reserve first request: %v", err)
	}
	if err := manager.Settle(ctx, "tester", date, 2, 2); err != nil {
		t.Fatalf("Settle: %v", err)
	}
	if _, err := manager.Reserve(ctx, "tester", 1.01, 10, 3); err == nil {
		t.Fatal("Reserve must reject a request whose estimate crosses the monthly limit")
	}
}

func TestReserveExceededErrorCarriesDailyResetTime(t *testing.T) {
	manager, _ := newTestManager(t)
	ctx := context.Background()
	before := time.Now().UTC()
	_, err := manager.Reserve(ctx, "tester", 6, 5, 20)
	var exceeded *ExceededError
	if !errors.As(err, &exceeded) {
		t.Fatalf("Reserve error = %v (%T), want *ExceededError", err, err)
	}
	if exceeded.Scope != "daily" {
		t.Fatalf("Scope = %q, want %q", exceeded.Scope, "daily")
	}
	if exceeded.Limit != 5 || exceeded.Estimate != 6 {
		t.Fatalf("Limit = %.2f, Estimate = %.2f, want 5.00 and 6.00", exceeded.Limit, exceeded.Estimate)
	}
	wantReset := nextUTCMidnight(before)
	if !exceeded.ResetAt.Equal(wantReset) {
		t.Fatalf("ResetAt = %v, want %v", exceeded.ResetAt, wantReset)
	}
	if exceeded.ResetAt.Before(before) {
		t.Fatalf("ResetAt %v must be in the future relative to %v", exceeded.ResetAt, before)
	}
}

func TestReserveExceededErrorCarriesMonthlyResetTime(t *testing.T) {
	manager, _ := newTestManager(t)
	ctx := context.Background()
	before := time.Now().UTC()
	_, err := manager.Reserve(ctx, "tester", 21, 0, 20)
	var exceeded *ExceededError
	if !errors.As(err, &exceeded) {
		t.Fatalf("Reserve error = %v (%T), want *ExceededError", err, err)
	}
	if exceeded.Scope != "monthly" {
		t.Fatalf("Scope = %q, want %q", exceeded.Scope, "monthly")
	}
	wantReset := nextUTCMonth(before)
	if !exceeded.ResetAt.Equal(wantReset) {
		t.Fatalf("ResetAt = %v, want %v", exceeded.ResetAt, wantReset)
	}
}

func TestExceededErrorCustomerMessageHasNoDollarAmount(t *testing.T) {
	e := &ExceededError{Scope: "monthly", Used: 3.7708, Estimate: 1.2798, Limit: 5, ResetAt: time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)}
	msg := e.CustomerMessage()
	if got, want := msg, "This request would exceed the monthly usage limit. Resets 2026-08-01 00:00 UTC."; got != want {
		t.Fatalf("CustomerMessage = %q, want %q", got, want)
	}
	if strings.Contains(msg, "$") {
		t.Fatalf("CustomerMessage = %q, must not name a dollar amount", msg)
	}
	// Error() stays available for logs with full reservation detail.
	if e.Error() == msg {
		t.Fatalf("Error() and CustomerMessage() should differ (log detail vs customer-facing text)")
	}
}

func TestReserveIsolatesSeparateUsers(t *testing.T) {
	manager, _ := newTestManager(t)
	ctx := context.Background()
	if _, err := manager.Reserve(ctx, "user-a", 5, 5, 20); err != nil {
		t.Fatalf("user-a Reserve: %v", err)
	}
	// user-a is now at its daily limit; user-b must have its own untouched budget.
	if _, err := manager.Reserve(ctx, "user-a", 0.01, 5, 20); err == nil {
		t.Fatal("user-a Reserve must be rejected once its own daily limit is hit")
	}
	if _, err := manager.Reserve(ctx, "user-b", 5, 5, 20); err != nil {
		t.Fatalf("user-b Reserve must succeed against its own untouched budget, got: %v", err)
	}
}

func TestReserveSurvivesStoreRestart(t *testing.T) {
	ctx := context.Background()
	dbPath := filepath.Join(t.TempDir(), "telemetry.db")

	store, err := telemetry.NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	manager := NewManager(store)
	date, err := manager.Reserve(ctx, "tester", 4.50, 5, 20)
	if err != nil {
		t.Fatalf("Reserve before restart: %v", err)
	}
	if err := manager.Settle(ctx, "tester", date, 4.50, 4.50); err != nil {
		t.Fatalf("Settle before restart: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Simulate a process restart: reopen the same on-disk store and rebuild the Manager.
	restarted, err := telemetry.NewStore(dbPath)
	if err != nil {
		t.Fatalf("NewStore after restart: %v", err)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	restartedManager := NewManager(restarted)

	if _, err := restartedManager.Reserve(ctx, "tester", 0.51, 5, 20); err == nil {
		t.Fatal("usage recorded before restart must still count against the limit after restart")
	}
	usage, err := restarted.GetDailyUsage(ctx, budgetStorageKey("tester"), time.Now().UTC().Format("2006-01-02"))
	if err != nil {
		t.Fatalf("GetDailyUsage after restart: %v", err)
	}
	if usage != 4.50 {
		t.Fatalf("daily usage after restart = %.2f, want 4.50", usage)
	}
}

func TestReserveRolloverAcrossDayAndMonthBoundaries(t *testing.T) {
	manager, store := newTestManager(t)
	ctx := context.Background()
	now := time.Now().UTC()

	yesterday := now.AddDate(0, 0, -1).Format("2006-01-02")
	// AddDate on `now` directly can normalize back into the current month (a
	// month subtracted from e.g. Mar 31 lands on Mar 3, not Feb), silently
	// skipping the monthly-boundary case it's meant to exercise. Anchoring to
	// the 1st of the month first is safe on every date, since day 1 never
	// overflows a month subtraction.
	firstOfThisMonth := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.UTC)
	lastMonth := firstOfThisMonth.AddDate(0, -1, 0).Format("2006-01")
	if lastMonth == now.Format("2006-01") {
		t.Fatalf("test bug: computed last month %q equals the current month", lastMonth)
	}
	storageID := budgetStorageKey("tester")

	// Seed spend that landed on a prior day and, separately, a prior calendar
	// month, both against the SAME api key. Neither must count toward today's
	// daily limit or this month's monthly limit.
	if err := store.RecordBudgetUsage(ctx, storageID, yesterday, 4.99); err != nil {
		t.Fatalf("seed yesterday's usage: %v", err)
	}
	if err := store.RecordBudgetUsage(ctx, storageID, lastMonth+"-01", 19.99); err != nil {
		t.Fatalf("seed last month's usage: %v", err)
	}

	if _, err := manager.Reserve(ctx, "tester", 5, 5, 20); err != nil {
		t.Fatalf("today's Reserve must not be blocked by a prior day's/month's usage: %v", err)
	}
}

func TestExceededErrorCustomerMessageNamesTheScopeThatWasHit(t *testing.T) {
	daily := &ExceededError{Scope: "daily", Limit: 3, ResetAt: time.Date(2026, 7, 31, 0, 0, 0, 0, time.UTC)}
	if got, want := daily.CustomerMessage(), "This request would exceed the daily usage limit. Resets 2026-07-31 00:00 UTC."; got != want {
		t.Fatalf("CustomerMessage = %q, want %q", got, want)
	}
	monthly := &ExceededError{Scope: "monthly", Limit: 90, ResetAt: time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)}
	if got, want := monthly.CustomerMessage(), "This request would exceed the monthly usage limit. Resets 2026-08-01 00:00 UTC."; got != want {
		t.Fatalf("CustomerMessage = %q, want %q", got, want)
	}
}
