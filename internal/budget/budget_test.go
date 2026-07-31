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
