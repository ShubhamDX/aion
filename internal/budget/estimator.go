package budget

import (
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/types"
)

// Estimator provides pre-routing cost estimates.
type Estimator struct {
	table *pricing.Table
}

// NewEstimator creates an Estimator backed by the given pricing table.
func NewEstimator(table *pricing.Table) *Estimator {
	return &Estimator{table: table}
}

// EstimateInputTokens estimates token count from a request using a bytes/4
// heuristic. Tool call arguments and tool schemas are counted by their actual
// size rather than a flat per-tool estimate, since either can dominate a
// request's real size.
func EstimateInputTokens(req *types.ChatCompletionRequest) int {
	totalChars := 0
	for _, msg := range req.Messages {
		totalChars += len(msg.ContentString())
		totalChars += len(msg.Role) + 4 // role + formatting overhead
		for _, tc := range msg.ToolCalls {
			totalChars += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	for _, tool := range req.Tools {
		totalChars += len(tool.Function.Name) + len(tool.Function.Description) + len(tool.Function.Parameters)
	}
	return totalChars / 4
}

// Estimate returns estimated cost for a model given an estimated input token count.
// Assumes output tokens ~= input tokens * 1.5 for estimation purposes.
func (e *Estimator) Estimate(modelID string, inputTokens int) float64 {
	estimatedOutputTokens := int(float64(inputTokens) * 1.5)
	return e.table.EstimateCost(modelID, inputTokens, estimatedOutputTokens)
}

// EstimateMaxTierCost estimates the cost if the most expensive model were used.
// Used for savings calculation.
func (e *Estimator) EstimateMaxTierCost(inputTokens, outputTokens int) float64 {
	return e.table.MostExpensiveModelCost(inputTokens, outputTokens)
}
