package telemetry

import (
	"context"
	"fmt"
)

// SavingsReport aggregates cost and savings data over a time range.
type SavingsReport struct {
	TotalCostUSD            float64
	TotalSavingsUSD         float64
	TotalRequests           float64
	EffectiveSavingsPercent float64
}

// QuerySavings returns an aggregate SavingsReport for requests whose
// created_at falls within [from, to]. Both bounds are inclusive and should
// be formatted as RFC 3339 or YYYY-MM-DD strings.
func (s *Store) QuerySavings(ctx context.Context, from, to string) (*SavingsReport, error) {
	const q = `SELECT
		COALESCE(SUM(cost_usd), 0),
		COALESCE(SUM(savings_usd), 0),
		COUNT(*)
		FROM request_log
		WHERE created_at >= ? AND created_at <= ?`

	var r SavingsReport
	if err := s.db.QueryRowContext(ctx, q, from, to).Scan(
		&r.TotalCostUSD,
		&r.TotalSavingsUSD,
		&r.TotalRequests,
	); err != nil {
		return nil, fmt.Errorf("telemetry: query savings: %w", err)
	}

	total := r.TotalCostUSD + r.TotalSavingsUSD
	if total > 0 {
		r.EffectiveSavingsPercent = (r.TotalSavingsUSD / total) * 100
	}

	return &r, nil
}

// RoutingStats describes request volume and latency for a tier+model pair.
type RoutingStats struct {
	Tier         int
	Model        string
	Count        int
	AvgLatencyMS float64
}

// QueryRoutingDistribution returns per-tier/model routing statistics for the
// given time range.
func (s *Store) QueryRoutingDistribution(ctx context.Context, from, to string) ([]RoutingStats, error) {
	const q = `SELECT tier, model, COUNT(*) AS cnt, AVG(latency_ms) AS avg_lat
		FROM request_log
		WHERE created_at >= ? AND created_at <= ?
		GROUP BY tier, model
		ORDER BY tier, cnt DESC`

	rows, err := s.db.QueryContext(ctx, q, from, to)
	if err != nil {
		return nil, fmt.Errorf("telemetry: query routing distribution: %w", err)
	}
	defer rows.Close()

	var results []RoutingStats
	for rows.Next() {
		var rs RoutingStats
		if err := rows.Scan(&rs.Tier, &rs.Model, &rs.Count, &rs.AvgLatencyMS); err != nil {
			return nil, fmt.Errorf("telemetry: scan routing stats: %w", err)
		}
		results = append(results, rs)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("telemetry: rows iteration: %w", err)
	}
	return results, nil
}

// CostBreakdown shows per-day, per-model cost and request count.
type CostBreakdown struct {
	Date     string
	Model    string
	CostUSD  float64
	Requests int
}

// QueryCostBreakdown returns daily cost breakdowns by model for the given
// time range.
func (s *Store) QueryCostBreakdown(ctx context.Context, from, to string) ([]CostBreakdown, error) {
	const q = `SELECT
		DATE(created_at) AS dt,
		model,
		SUM(cost_usd) AS total_cost,
		COUNT(*) AS cnt
		FROM request_log
		WHERE created_at >= ? AND created_at <= ?
		GROUP BY dt, model
		ORDER BY dt, total_cost DESC`

	rows, err := s.db.QueryContext(ctx, q, from, to)
	if err != nil {
		return nil, fmt.Errorf("telemetry: query cost breakdown: %w", err)
	}
	defer rows.Close()

	var results []CostBreakdown
	for rows.Next() {
		var cb CostBreakdown
		if err := rows.Scan(&cb.Date, &cb.Model, &cb.CostUSD, &cb.Requests); err != nil {
			return nil, fmt.Errorf("telemetry: scan cost breakdown: %w", err)
		}
		results = append(results, cb)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("telemetry: rows iteration: %w", err)
	}
	return results, nil
}
