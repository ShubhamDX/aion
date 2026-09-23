package budget

import (
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestEstimateInputTokensCountsToolCallArguments(t *testing.T) {
	small := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "assistant", ToolCalls: []types.ToolCall{{ID: "call_1", Type: "function", Function: types.FunctionCall{Name: "f", Arguments: `{}`}}}},
		},
	}
	large := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "assistant", ToolCalls: []types.ToolCall{{ID: "call_1", Type: "function", Function: types.FunctionCall{Name: "f", Arguments: `{"data":"` + strings.Repeat("x", 100_000) + `"}`}}}},
		},
	}
	smallEstimate := EstimateInputTokens(small)
	largeEstimate := EstimateInputTokens(large)
	if largeEstimate < smallEstimate+20_000 {
		t.Fatalf("large tool call arguments (~100KB) did not move the estimate: small=%d large=%d", smallEstimate, largeEstimate)
	}
}

func TestEstimateInputTokensCountsToolSchemaSize(t *testing.T) {
	small := &types.ChatCompletionRequest{
		Tools: []types.Tool{{Type: "function", Function: types.FunctionDef{Name: "f", Parameters: []byte(`{}`)}}},
	}
	large := &types.ChatCompletionRequest{
		Tools: []types.Tool{{Type: "function", Function: types.FunctionDef{Name: "f", Parameters: []byte(`{"schema":"` + strings.Repeat("x", 100_000) + `"}`)}}},
	}
	smallEstimate := EstimateInputTokens(small)
	largeEstimate := EstimateInputTokens(large)
	if largeEstimate < smallEstimate+20_000 {
		t.Fatalf("large tool schema (~100KB) did not move the estimate: small=%d large=%d", smallEstimate, largeEstimate)
	}
}
