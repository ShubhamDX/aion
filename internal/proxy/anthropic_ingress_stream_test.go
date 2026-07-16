package proxy

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestAnthropicSSEStateKeepsFragmentedToolCallInOneBlock(t *testing.T) {
	recorder := httptest.NewRecorder()
	var state anthropicSSEState
	toolIndex := 0

	state.writeChoice(recorder, recorder, types.ChunkChoice{
		Delta: types.ChunkDelta{ToolCalls: []types.ToolCall{{
			Index: &toolIndex,
			ID:    "toolu_1",
			Type:  "function",
			Function: types.FunctionCall{
				Name: "Bash",
			},
		}}},
	})
	for _, arguments := range []string{`{"command":`, `"printf ok"}`} {
		state.writeChoice(recorder, recorder, types.ChunkChoice{
			Delta: types.ChunkDelta{ToolCalls: []types.ToolCall{{
				Index: &toolIndex,
				ID:    "toolu_1",
				Type:  "function",
				Function: types.FunctionCall{
					Name:      "Bash",
					Arguments: arguments,
				},
			}}},
		})
	}
	state.closeBlock(recorder, recorder)

	body := recorder.Body.String()
	if got := strings.Count(body, `"type":"tool_use"`); got != 1 {
		t.Fatalf("tool_use block starts = %d, want 1\n%s", got, body)
	}
	if got := strings.Count(body, "event: content_block_stop"); got != 1 {
		t.Fatalf("content block stops = %d, want 1\n%s", got, body)
	}
	if got := strings.Count(body, `"type":"input_json_delta"`); got != 2 {
		t.Fatalf("input JSON deltas = %d, want 2\n%s", got, body)
	}
	if !strings.Contains(body, `"partial_json":"{\"command\":"`) ||
		!strings.Contains(body, `"partial_json":"\"printf ok\"}"`) {
		t.Fatalf("fragmented arguments were not preserved\n%s", body)
	}
}
