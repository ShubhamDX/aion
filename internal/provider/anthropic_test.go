package provider

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/pkg/types"
)

// SP2b-2 golden test for a TRANSLATED provider: carrying schema settings must not
// change the Anthropic upstream payload. translateRequest builds its own struct
// field-by-field and never reads SchemaSettings, so the bytes must match.
func TestAnthropicTranslate_SchemaSettingsNeverLeak(t *testing.T) {
	p := &AnthropicProvider{}
	base := &types.ChatCompletionRequest{
		Model:    "claude-x",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	withCarrier := *base
	withCarrier.SchemaSettings = &types.SchemaSettings{
		Mode: "tool_schema", SchemaPolicyID: "sp-orders", SchemaVersion: "1",
		SchemaHash: "deadbeef", FailClosed: true, Reason: "tool_schema_supported",
	}

	bare, err := json.Marshal(p.translateRequest(base, "claude-3", false))
	if err != nil {
		t.Fatalf("marshal bare: %v", err)
	}
	carrier, err := json.Marshal(p.translateRequest(&withCarrier, "claude-3", false))
	if err != nil {
		t.Fatalf("marshal carrier: %v", err)
	}
	if string(bare) != string(carrier) {
		t.Fatalf("schema settings changed the Anthropic payload:\n bare:    %s\n carrier: %s", bare, carrier)
	}
	for _, banned := range []string{"tool_schema", "sp-orders", "deadbeef", "schema"} {
		if strings.Contains(string(carrier), banned) {
			t.Fatalf("schema setting %q leaked into Anthropic payload: %s", banned, carrier)
		}
	}
}

func TestUsageFromAnthropicIncludesCacheTokens(t *testing.T) {
	usage := usageFromAnthropic(anthropicUsage{
		InputTokens:                40,
		CacheReadInputTokens:       50,
		CacheCreationInputTokens:   9,
		CacheCreationInputTokens1h: 1,
		OutputTokens:               20,
		ServiceTier:                "ephemeral",
	})
	if usage.PromptTokens != 100 || usage.CompletionTokens != 20 || usage.TotalTokens != 120 {
		t.Fatalf("bad aggregate usage: %+v", usage)
	}
	if usage.UncachedInputTokens != 40 || usage.CacheReadInputTokens != 50 || usage.CacheCreationInputTokens != 10 {
		t.Fatalf("bad cache partition: %+v", usage)
	}
	if usage.ProviderCacheMode != "ephemeral" {
		t.Fatalf("cache mode = %q", usage.ProviderCacheMode)
	}
	if !usage.InputPartitionValid() {
		t.Fatal("usage must satisfy partition invariant")
	}
}

func TestUsageFromAnthropicOmitsPartitionWithoutCache(t *testing.T) {
	usage := usageFromAnthropic(anthropicUsage{InputTokens: 40, OutputTokens: 20})
	if usage.PromptTokens != 40 || usage.CompletionTokens != 20 || usage.TotalTokens != 60 {
		t.Fatalf("bad aggregate usage: %+v", usage)
	}
	if usage.UncachedInputTokens != 0 || usage.CacheReadInputTokens != 0 || usage.CacheCreationInputTokens != 0 {
		t.Fatalf("cache fields should stay omitted without cache: %+v", usage)
	}
}

func TestTranslateAnthropicMessages_OpenAIToolConversation(t *testing.T) {
	p := &AnthropicProvider{}
	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "system", Content: json.RawMessage(`"use tools carefully"`)},
			{Role: "user", Content: json.RawMessage(`"check the task"`)},
			{
				Role:    "assistant",
				Content: json.RawMessage(`""`),
				ToolCalls: []types.ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: types.FunctionCall{
							Name:      "kanban_sh",
							Arguments: `{"args":["show","t_1"]}`,
						},
					},
				},
			},
			{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage(`"task details"`)},
		},
	}

	payload := p.translateRequest(req, "claude-test", false)
	if payload.System != "use tools carefully" {
		t.Fatalf("system = %q", payload.System)
	}
	if len(payload.Messages) != 3 {
		t.Fatalf("messages len = %d, want 3: %+v", len(payload.Messages), payload.Messages)
	}
	if payload.Messages[1].Role != "assistant" {
		t.Fatalf("assistant message role = %q", payload.Messages[1].Role)
	}
	assistantBlocks, ok := payload.Messages[1].Content.([]anthropicContent)
	if !ok || len(assistantBlocks) != 1 {
		t.Fatalf("assistant content = %#v", payload.Messages[1].Content)
	}
	if assistantBlocks[0].Type != "tool_use" || assistantBlocks[0].ID != "call_1" ||
		assistantBlocks[0].Name != "kanban_sh" || string(assistantBlocks[0].Input) != `{"args":["show","t_1"]}` {
		t.Fatalf("bad tool_use block: %+v", assistantBlocks[0])
	}
	userBlocks, ok := payload.Messages[2].Content.([]anthropicContent)
	if !ok || len(userBlocks) != 1 {
		t.Fatalf("tool result content = %#v", payload.Messages[2].Content)
	}
	if payload.Messages[2].Role != "user" || userBlocks[0].Type != "tool_result" ||
		userBlocks[0].ToolUseID != "call_1" || userBlocks[0].Content != "task details" {
		t.Fatalf("bad tool_result message: role=%q blocks=%+v", payload.Messages[2].Role, userBlocks)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(body), `"role":"tool"`) {
		t.Fatalf("Anthropic-family payload must not contain role=tool: %s", body)
	}
}

func TestTranslateAnthropicMessages_GroupsAdjacentToolResults(t *testing.T) {
	_, messages := translateAnthropicMessages([]types.Message{
		{Role: "assistant", ToolCalls: []types.ToolCall{
			{ID: "call_1", Type: "function", Function: types.FunctionCall{Name: "first", Arguments: `{}`}},
			{ID: "call_2", Type: "function", Function: types.FunctionCall{Name: "second", Arguments: `{}`}},
		}},
		{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage(`"one"`)},
		{Role: "tool", ToolCallID: "call_2", Content: json.RawMessage(`"two"`)},
	})

	if len(messages) != 2 {
		t.Fatalf("messages len = %d, want 2: %+v", len(messages), messages)
	}
	blocks, ok := messages[1].Content.([]anthropicContent)
	if !ok || len(blocks) != 2 {
		t.Fatalf("grouped tool results = %#v", messages[1].Content)
	}
	if blocks[0].ToolUseID != "call_1" || blocks[1].ToolUseID != "call_2" {
		t.Fatalf("tool result ids = %+v", blocks)
	}
}

func TestTranslateAnthropicMessages_UnboundToolMessageBecomesUserText(t *testing.T) {
	_, messages := translateAnthropicMessages([]types.Message{
		{Role: "tool", Content: json.RawMessage(`"legacy context"`)},
	})
	if len(messages) != 1 || messages[0].Role != "user" || messages[0].Content != "legacy context" {
		t.Fatalf("unbound tool message translation = %+v", messages)
	}
}

func TestBedrockTranslate_OpenAIToolConversationHasNoToolRole(t *testing.T) {
	p := &BedrockProvider{}
	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"start"`)},
			{Role: "assistant", ToolCalls: []types.ToolCall{
				{ID: "call_1", Type: "function", Function: types.FunctionCall{Name: "lookup", Arguments: `{"q":"x"}`}},
			}},
			{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage(`"result"`)},
		},
	}
	body, err := json.Marshal(p.translateRequest(req, false))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(body), `"role":"tool"`) {
		t.Fatalf("Bedrock payload must not contain role=tool: %s", body)
	}
	for _, want := range []string{`"type":"tool_use"`, `"type":"tool_result"`, `"tool_use_id":"call_1"`} {
		if !strings.Contains(string(body), want) {
			t.Fatalf("Bedrock payload missing %s: %s", want, body)
		}
	}
}

func TestAnthropicStreamToolUseEmitsOpenAIToolCallDeltas(t *testing.T) {
	s := &anthropicStreamReader{id: "msg_1", model: "claude-test"}

	start := &anthropicStreamEvent{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: &anthropicContent{
			Type: "tool_use",
			ID:   "call_1",
			Name: "lookup",
		},
	}
	startChunk, blocks := translateAnthropicContentBlockStart(s.id, s.model, s.toolBlocks, start)
	s.toolBlocks = blocks
	if startChunk == nil || len(startChunk.Choices[0].Delta.ToolCalls) != 1 {
		t.Fatalf("start chunk = %+v", startChunk)
	}
	startCall := startChunk.Choices[0].Delta.ToolCalls[0]
	if startCall.Index == nil || *startCall.Index != 0 || startCall.ID != "call_1" ||
		startCall.Type != "function" || startCall.Function.Name != "lookup" {
		t.Fatalf("bad start tool call delta: %+v", startCall)
	}

	delta := &anthropicStreamEvent{
		Type:  "content_block_delta",
		Index: 0,
		Delta: json.RawMessage(`{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}`),
	}
	deltaChunk, err := translateAnthropicContentBlockDelta(s.id, s.model, s.toolBlocks, delta)
	if err != nil {
		t.Fatalf("delta: %v", err)
	}
	if deltaChunk == nil || len(deltaChunk.Choices[0].Delta.ToolCalls) != 1 {
		t.Fatalf("delta chunk = %+v", deltaChunk)
	}
	deltaCall := deltaChunk.Choices[0].Delta.ToolCalls[0]
	if deltaCall.Index == nil || *deltaCall.Index != 0 || deltaCall.ID != "call_1" ||
		deltaCall.Function.Name != "lookup" || deltaCall.Function.Arguments != `{"q":"x"}` {
		t.Fatalf("bad argument delta: %+v", deltaCall)
	}
}
