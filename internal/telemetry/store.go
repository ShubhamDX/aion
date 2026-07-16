package telemetry

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite"
)

// RequestEvent represents a single proxied LLM request for telemetry.
// Defined locally until internal/types is finalized.
type RequestEvent struct {
	RequestID    string
	APIKeyID     string
	Tier         int
	Model        string
	Provider     string
	InputTokens  int
	OutputTokens int
	CostUSD      float64
	SavingsUSD   float64
	LatencyMS    int64
	StatusCode   int
	Stream       bool
	CreatedAt    time.Time
}

// Store is the SQLite-backed telemetry storage layer.
type Store struct {
	db *sql.DB
}

// initSQL contains the schema migration executed on startup.
const initSQL = `
CREATE TABLE IF NOT EXISTS request_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    request_id  TEXT    NOT NULL,
    api_key_id  TEXT    NOT NULL,
    tier        INTEGER NOT NULL,
    model       TEXT    NOT NULL,
    provider    TEXT    NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0.0,
    savings_usd     REAL    NOT NULL DEFAULT 0.0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    status_code     INTEGER NOT NULL DEFAULT 200,
    stream          BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_request_log_created_at ON request_log(created_at);
CREATE INDEX IF NOT EXISTS idx_request_log_api_key_id ON request_log(api_key_id);
CREATE INDEX IF NOT EXISTS idx_request_log_model ON request_log(model);

CREATE TABLE IF NOT EXISTS budget_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id  TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(api_key_id, date)
);

CREATE INDEX IF NOT EXISTS idx_budget_usage_key_date ON budget_usage(api_key_id, date);
`

// NewStore opens a SQLite database at dbPath, runs migrations, and returns a Store.
// It enables WAL mode and sets a busy timeout for concurrent access.
func NewStore(dbPath string) (*Store, error) {
	// Ensure the parent directory exists.
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("telemetry: create db directory: %w", err)
	}

	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("telemetry: open db: %w", err)
	}

	// Limit connections — SQLite serialises writes anyway.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("telemetry: ping db: %w", err)
	}

	if _, err := db.Exec(initSQL); err != nil {
		db.Close()
		return nil, fmt.Errorf("telemetry: run migrations: %w", err)
	}

	return &Store{db: db}, nil
}

// InsertEvent writes a single RequestEvent to the request_log table.
func (s *Store) InsertEvent(ctx context.Context, event RequestEvent) error {
	const q = `INSERT INTO request_log
		(request_id, api_key_id, tier, model, provider,
		 input_tokens, output_tokens, cost_usd, savings_usd,
		 latency_ms, status_code, stream, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := s.db.ExecContext(ctx, q,
		event.RequestID,
		event.APIKeyID,
		event.Tier,
		event.Model,
		event.Provider,
		event.InputTokens,
		event.OutputTokens,
		event.CostUSD,
		event.SavingsUSD,
		event.LatencyMS,
		event.StatusCode,
		event.Stream,
		event.CreatedAt.UTC().Format(time.RFC3339Nano),
	)
	if err != nil {
		return fmt.Errorf("telemetry: insert event: %w", err)
	}
	return nil
}

// InsertEvents writes a batch of RequestEvents inside a single transaction.
func (s *Store) InsertEvents(ctx context.Context, events []RequestEvent) error {
	if len(events) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("telemetry: begin tx: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, `INSERT INTO request_log
		(request_id, api_key_id, tier, model, provider,
		 input_tokens, output_tokens, cost_usd, savings_usd,
		 latency_ms, status_code, stream, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("telemetry: prepare stmt: %w", err)
	}
	defer stmt.Close()

	for _, e := range events {
		if _, err := stmt.ExecContext(ctx, e.RequestID, e.APIKeyID, e.Tier, e.Model, e.Provider,
			e.InputTokens, e.OutputTokens, e.CostUSD, e.SavingsUSD,
			e.LatencyMS, e.StatusCode, e.Stream,
			e.CreatedAt.UTC().Format(time.RFC3339Nano)); err != nil {
			return fmt.Errorf("telemetry: insert event batch: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("telemetry: commit tx: %w", err)
	}
	return nil
}

// RecordBudgetUsage upserts a cost entry into the budget_usage table for the
// given API key and date (YYYY-MM-DD). If a row already exists it increments
// the cost.
func (s *Store) RecordBudgetUsage(ctx context.Context, apiKeyID string, date string, cost float64) error {
	const q = `INSERT INTO budget_usage (api_key_id, date, cost_usd)
		VALUES (?, ?, ?)
		ON CONFLICT(api_key_id, date) DO UPDATE SET cost_usd = cost_usd + excluded.cost_usd`

	if _, err := s.db.ExecContext(ctx, q, apiKeyID, date, cost); err != nil {
		return fmt.Errorf("telemetry: record budget usage: %w", err)
	}
	return nil
}

// MigrateBudgetUsageKey moves legacy rows from a raw API-key identifier to a
// non-secret stable identifier. Existing rows for the destination are merged.
func (s *Store) MigrateBudgetUsageKey(ctx context.Context, oldID, newID string) error {
	if oldID == "" || newID == "" || oldID == newID {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("telemetry: begin budget key migration: %w", err)
	}
	defer tx.Rollback()

	const merge = `INSERT INTO budget_usage (api_key_id, date, cost_usd)
		SELECT ?, date, SUM(cost_usd) FROM budget_usage WHERE api_key_id = ? GROUP BY date
		ON CONFLICT(api_key_id, date) DO UPDATE SET cost_usd = cost_usd + excluded.cost_usd`
	if _, err := tx.ExecContext(ctx, merge, newID, oldID); err != nil {
		return fmt.Errorf("telemetry: merge budget key rows: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM budget_usage WHERE api_key_id = ?`, oldID); err != nil {
		return fmt.Errorf("telemetry: delete legacy budget key rows: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("telemetry: commit budget key migration: %w", err)
	}
	return nil
}

// ReserveBudgetUsage atomically checks the projected daily and monthly usage,
// then adds the estimate to the daily row. Serializing this transaction with
// the store's single SQLite connection prevents concurrent requests from both
// passing the same remaining budget check.
func (s *Store) ReserveBudgetUsage(
	ctx context.Context,
	apiKeyID string,
	date string,
	yearMonth string,
	estimate float64,
	dailyLimit float64,
	monthlyLimit float64,
) (dailyUsage float64, monthlyUsage float64, exceeded string, err error) {
	if estimate < 0 {
		return 0, 0, "", fmt.Errorf("telemetry: budget estimate must not be negative")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, 0, "", fmt.Errorf("telemetry: begin budget reservation: %w", err)
	}
	defer tx.Rollback()

	const dailyQuery = `SELECT COALESCE(cost_usd, 0) FROM budget_usage WHERE api_key_id = ? AND date = ?`
	if err := tx.QueryRowContext(ctx, dailyQuery, apiKeyID, date).Scan(&dailyUsage); err != nil && err != sql.ErrNoRows {
		return 0, 0, "", fmt.Errorf("telemetry: get daily usage for reservation: %w", err)
	}

	const monthlyQuery = `SELECT COALESCE(SUM(cost_usd), 0) FROM budget_usage
		WHERE api_key_id = ? AND date LIKE ? || '%'`
	if err := tx.QueryRowContext(ctx, monthlyQuery, apiKeyID, yearMonth).Scan(&monthlyUsage); err != nil {
		return 0, 0, "", fmt.Errorf("telemetry: get monthly usage for reservation: %w", err)
	}

	if dailyLimit > 0 && dailyUsage+estimate > dailyLimit {
		return dailyUsage, monthlyUsage, "daily", nil
	}
	if monthlyLimit > 0 && monthlyUsage+estimate > monthlyLimit {
		return dailyUsage, monthlyUsage, "monthly", nil
	}

	if estimate > 0 {
		const reserveQuery = `INSERT INTO budget_usage (api_key_id, date, cost_usd)
			VALUES (?, ?, ?)
			ON CONFLICT(api_key_id, date) DO UPDATE SET cost_usd = cost_usd + excluded.cost_usd`
		if _, err := tx.ExecContext(ctx, reserveQuery, apiKeyID, date, estimate); err != nil {
			return 0, 0, "", fmt.Errorf("telemetry: reserve budget usage: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return 0, 0, "", fmt.Errorf("telemetry: commit budget reservation: %w", err)
	}
	return dailyUsage, monthlyUsage, "", nil
}

// AdjustBudgetUsage replaces a prior estimate with actual cost. A process crash
// can leave an estimate in place, which fails closed by reducing the remaining
// budget until an operator reconciles the usage row.
func (s *Store) AdjustBudgetUsage(ctx context.Context, apiKeyID string, date string, delta float64) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("telemetry: begin budget adjustment: %w", err)
	}
	defer tx.Rollback()

	result, err := tx.ExecContext(ctx,
		`UPDATE budget_usage SET cost_usd = MAX(cost_usd + ?, 0) WHERE api_key_id = ? AND date = ?`,
		delta, apiKeyID, date,
	)
	if err != nil {
		return fmt.Errorf("telemetry: adjust budget usage: %w", err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("telemetry: inspect budget adjustment: %w", err)
	}
	if rows == 0 {
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO budget_usage (api_key_id, date, cost_usd) VALUES (?, ?, MAX(?, 0))`,
			apiKeyID, date, delta,
		); err != nil {
			return fmt.Errorf("telemetry: insert budget adjustment: %w", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("telemetry: commit budget adjustment: %w", err)
	}
	return nil
}

// GetDailyUsage returns the total cost for an API key on a specific date (YYYY-MM-DD).
func (s *Store) GetDailyUsage(ctx context.Context, apiKeyID string, date string) (float64, error) {
	const q = `SELECT COALESCE(cost_usd, 0) FROM budget_usage WHERE api_key_id = ? AND date = ?`
	var cost float64
	if err := s.db.QueryRowContext(ctx, q, apiKeyID, date).Scan(&cost); err != nil {
		if err == sql.ErrNoRows {
			return 0, nil
		}
		return 0, fmt.Errorf("telemetry: get daily usage: %w", err)
	}
	return cost, nil
}

// GetMonthlyUsage returns the total cost for an API key in a month (YYYY-MM).
func (s *Store) GetMonthlyUsage(ctx context.Context, apiKeyID string, yearMonth string) (float64, error) {
	const q = `SELECT COALESCE(SUM(cost_usd), 0) FROM budget_usage
		WHERE api_key_id = ? AND date LIKE ? || '%'`
	var cost float64
	if err := s.db.QueryRowContext(ctx, q, apiKeyID, yearMonth).Scan(&cost); err != nil {
		if err == sql.ErrNoRows {
			return 0, nil
		}
		return 0, fmt.Errorf("telemetry: get monthly usage: %w", err)
	}
	return cost, nil
}

// Close closes the underlying database connection.
func (s *Store) Close() error {
	return s.db.Close()
}
