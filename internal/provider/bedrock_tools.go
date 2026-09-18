package provider

import (
	"encoding/json"
	"fmt"

	"github.com/ShubhamDX/aion/internal/types"
)

// Translate the caller's tool choice without weakening required or named calls.
// Requests without a choice retain the provider default.
func bedrockClaudeToolChoice(req *types.ChatCompletionRequest) (json.RawMessage, error) {
	choice := req.ToolChoice
	if len(choice) == 0 || string(choice) == "null" {
		return nil, nil
	}
	var mode string
	if err := json.Unmarshal(choice, &mode); err == nil {
		switch mode {
		case "none":
			return json.RawMessage(`{"type":"none"}`), nil
		case "auto", "required":
			if len(req.Tools) == 0 {
				return nil, fmt.Errorf("bedrock: tool_choice requires tools")
			}
			if mode == "required" {
				return json.RawMessage(`{"type":"any"}`), nil
			}
			return json.RawMessage(`{"type":"auto"}`), nil
		default:
			return nil, fmt.Errorf("bedrock: unsupported tool_choice")
		}
	}
	var named struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(choice, &named) != nil || named.Type != "function" || named.Function.Name == "" {
		return nil, fmt.Errorf("bedrock: invalid tool_choice")
	}
	for _, tool := range req.Tools {
		if tool.Type == "function" && tool.Function.Name == named.Function.Name {
			return json.Marshal(struct {
				Type string `json:"type"`
				Name string `json:"name"`
			}{"tool", named.Function.Name})
		}
	}
	return nil, fmt.Errorf("bedrock: tool_choice names an unavailable tool")
}
