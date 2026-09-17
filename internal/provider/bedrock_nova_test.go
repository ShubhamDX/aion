package provider

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"github.com/ShubhamDX/aion/internal/types"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNovaNativeToolsCacheAndUsage(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/model/us.amazon.nova-2-lite-v1:0/converse" || r.Header.Get("Authorization") != "Bearer test-only" {
			t.Error("wrong route or auth")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		system := body["system"].([]any)
		if len(system) != 2 || system[1].(map[string]any)["cachePoint"] == nil {
			t.Error("cache checkpoint lost")
		}
		msgs := body["messages"].([]any)
		if len(msgs) != 3 {
			t.Fatalf("tool history order changed: %v", msgs)
		}
		content := msgs[2].(map[string]any)["content"].([]any)
		if len(content) != 2 {
			t.Fatal("tool result and follow-up were not preserved")
		}
		w.Header().Set("X-Amzn-Requestid", "native-test")
		w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"result"},{"toolUse":{"toolUseId":"c2","name":"lookup","input":{"id":2}}}]}},"stopReason":"tool_use","usage":{"inputTokens":15,"outputTokens":16,"cacheWriteInputTokens":1488,"cacheWriteInputTokenCount":1488,"totalTokens":1519}}`))
	}))
	defer upstream.Close()
	p := &BedrockProvider{baseURL: upstream.URL, client: upstream.Client(), bearerToken: "test-only"}
	req := &types.ChatCompletionRequest{Messages: []types.Message{
		{Role: "system", Content: json.RawMessage(`[{"type":"text","text":"prefix","cache_control":{"type":"ephemeral"}}]`)},
		{Role: "user", Content: json.RawMessage(`"lookup"`)},
		{Role: "assistant", Content: json.RawMessage(`null`), ToolCalls: []types.ToolCall{{ID: "c1", Type: "function", Function: types.FunctionCall{Name: "lookup", Arguments: `{"id":1}`}}}},
		{Role: "tool", ToolCallID: "c1", Content: json.RawMessage(`"one"`)},
		{Role: "user", Content: json.RawMessage(`"continue"`)},
	}, Tools: []types.Tool{{Type: "function", Function: types.FunctionDef{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}}}, ToolChoice: json.RawMessage(`"auto"`)}
	resp, err := p.Send(context.Background(), req, "us.amazon.nova-2-lite-v1:0")
	if err != nil {
		t.Fatal(err)
	}
	u := resp.ChatResponse.Usage
	if u.PromptTokens != 1503 || u.TotalTokens != 1519 || u.CacheCreationInputTokens != 1488 || !u.InputPartitionValid() {
		t.Fatalf("cache counters double counted: %+v", u)
	}
	if resp.ChatResponse.ID != "native-test" || resp.ChatResponse.Choices[0].FinishReason != "tool_calls" || len(resp.ChatResponse.Choices[0].Message.ToolCalls) != 1 {
		t.Fatal("lost native output")
	}
}

func TestNovaRejectsUnsupportedContractsBeforeNetwork(t *testing.T) {
	p := &BedrockProvider{}
	for _, req := range []*types.ChatCompletionRequest{
		{ResponseFormat: &types.ResponseFormat{Type: "json_schema"}},
		{Messages: []types.Message{{Role: "user", Content: json.RawMessage(`[{"type":"image_url","image_url":{"url":"data:image/png;base64,x"}}]`)}}},
		{Messages: []types.Message{{Role: "assistant", Content: json.RawMessage(`[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]`)}}},
		{Messages: []types.Message{{Role: "tool", Content: json.RawMessage(`"missing id"`)}}},
	} {
		if _, err := p.Send(context.Background(), req, "amazon.nova-2-lite-v1:0"); err == nil {
			t.Fatal("unsupported contract silently dropped")
		}
	}

	if _, err := p.Send(context.Background(), &types.ChatCompletionRequest{ResponseFormat: &types.ResponseFormat{Type: "json_schema"}}, "us.anthropic.claude-opus-5"); err == nil {
		t.Fatal("Claude schema silently dropped")
	}
}

func TestNovaUsagePartitions(t *testing.T) {
	for _, u := range []novaUsage{{Input: 10, Output: 5, Total: 15}, {Input: 10, Output: 5, ReadCount: 100, Total: 115}, {Input: 10, Output: 5, Write: 100, WriteCount: 100, Total: 115}} {
		got, err := u.normalized()
		if err != nil || !got.InputPartitionValid() {
			t.Fatalf("invalid partition: %+v %v", got, err)
		}
	}
	for _, u := range []novaUsage{{Input: 10, Read: 5, ReadCount: 6}, {Input: -1}, {Input: 10, Output: 5, Total: 10}} {
		if _, err := u.normalized(); err == nil {
			t.Fatal("invalid usage accepted")
		}
	}
}

func novaFrame(event, payload string) []byte {
	var headers bytes.Buffer
	for _, entry := range [][2]string{{":message-type", "event"}, {":event-type", event}} {
		headers.WriteByte(byte(len(entry[0])))
		headers.WriteString(entry[0])
		headers.WriteByte(7)
		binary.Write(&headers, binary.BigEndian, uint16(len(entry[1])))
		headers.WriteString(entry[1])
	}
	var frame bytes.Buffer
	binary.Write(&frame, binary.BigEndian, uint32(16+headers.Len()+len(payload)))
	binary.Write(&frame, binary.BigEndian, uint32(headers.Len()))
	binary.Write(&frame, binary.BigEndian, uint32(0))
	frame.Write(headers.Bytes())
	frame.WriteString(payload)
	binary.Write(&frame, binary.BigEndian, uint32(0))
	return frame.Bytes()
}

func TestNovaStreamTextToolsAndUsage(t *testing.T) {
	events := [][2]string{
		{"messageStart", `{"role":"assistant"}`},
		{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"hello"}}`},
		{"contentBlockStart", `{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"c1","name":"lookup"}}}`},
		{"contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"id\":1}"}}}`},
		{"messageStop", `{"stopReason":"tool_use"}`},
		{"metadata", `{"usage":{"inputTokens":10,"outputTokens":5,"cacheReadInputTokens":100,"totalTokens":115}}`},
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/model/amazon.nova-2-lite-v1:0/converse-stream" {
			t.Error("incorrect stream path")
		}
		for _, e := range events {
			w.Write(novaFrame(e[0], e[1]))
		}
	}))
	defer upstream.Close()
	p := &BedrockProvider{baseURL: upstream.URL, client: upstream.Client(), bearerToken: "test-only"}
	stream, err := p.SendStream(context.Background(), &types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hello"`)}}}, "amazon.nova-2-lite-v1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var chunks []*types.ChatCompletionChunk
	for {
		c, err := stream.ReadChunk()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, c)
	}
	if len(chunks) != 6 || *chunks[1].Choices[0].Delta.Content != "hello" || *chunks[2].Choices[0].Delta.ToolCalls[0].Index != 0 || chunks[3].Choices[0].Delta.ToolCalls[0].Function.Arguments != `{"id":1}` || chunks[5].Usage.CacheReadInputTokens != 100 {
		t.Fatalf("stream fields lost: %+v", chunks)
	}
}

func TestBedrockMalformedFrameLengthReturnsError(t *testing.T) {
	for _, length := range []uint32{0, 12, 32 * 1024 * 1024} {
		raw := make([]byte, 12)
		binary.BigEndian.PutUint32(raw, length)
		stream := bedrockStreamReader{reader: bytes.NewReader(raw)}
		if _, _, err := stream.readFrame(); err == nil {
			t.Fatal("invalid frame accepted")
		}
	}
}
