package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestBedrockClaudeToolChoiceOnBothEndpoints(t *testing.T) {
	for _, tc := range []struct{ input, want string }{
		{``, ``}, {`null`, ``}, {`"auto"`, `{"type":"auto"}`}, {`"required"`, `{"type":"any"}`},
		{`"none"`, `{"type":"none"}`}, {`{"type":"function","function":{"name":"submit_decision"}}`, `{"type":"tool","name":"submit_decision"}`},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(tc.input+map[bool]string{false: "/invoke", true: "/stream"}[stream], func(t *testing.T) {
				calls := 0
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls++
					var body map[string]json.RawMessage
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
						return
					}
					if string(body["tool_choice"]) != tc.want {
						t.Errorf("tool choice lost: got %s, want %s", body["tool_choice"], tc.want)
					}
					if body["tools"] == nil {
						t.Error("tools lost")
					}
					endpoint := "invoke"
					if stream {
						endpoint = "invoke-with-response-stream"
					}
					if r.URL.Path != "/model/us.anthropic.claude-sonnet-5/"+endpoint {
						t.Error("wrong endpoint")
					}
					if stream {
						return
					}
					io.WriteString(w, `{"id":"test","content":[{"type":"tool_use","id":"call1","name":"submit_decision","input":{"value":9007199254740993}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`)
				}))
				defer upstream.Close()
				p := &BedrockProvider{baseURL: upstream.URL, client: upstream.Client(), bearerToken: "test-only"}
				req := &types.ChatCompletionRequest{ToolChoice: json.RawMessage(tc.input), Tools: []types.Tool{{Type: "function", Function: types.FunctionDef{Name: "submit_decision", Parameters: json.RawMessage(`{"type":"object"}`)}}}, Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"decide"`)}}}
				before, _ := json.Marshal(req)
				if stream {
					reader, err := p.SendStream(context.Background(), req, "us.anthropic.claude-sonnet-5")
					if err != nil {
						t.Fatal(err)
					}
					reader.Close()
				} else {
					resp, err := p.Send(context.Background(), req, "us.anthropic.claude-sonnet-5")
					if err != nil {
						t.Fatal(err)
					}
					c := resp.ChatResponse.Choices[0]
					if c.FinishReason != "tool_calls" || c.Message.ToolCalls[0].Function.Arguments != `{"value":9007199254740993}` {
						t.Fatal("structured answer changed")
					}
				}
				after, _ := json.Marshal(req)
				if calls != 1 || string(before) != string(after) {
					t.Fatal("unexpected dispatch count or caller mutation")
				}
			})
		}
	}
}

func TestBedrockClaudeInvalidChoiceFailsBeforeNetwork(t *testing.T) {
	p := &BedrockProvider{}
	for _, choice := range []string{`"required"`, `"auto"`, `"unknown"`, `{}`, `{"type":"function","function":{"name":"missing"}}`} {
		req := &types.ChatCompletionRequest{ToolChoice: json.RawMessage(choice)}
		if _, err := p.Send(context.Background(), req, "us.anthropic.claude-sonnet-5"); err == nil {
			t.Fatal("invalid choice dispatched")
		}
		if _, err := p.SendStream(context.Background(), req, "us.anthropic.claude-sonnet-5"); err == nil {
			t.Fatal("invalid stream choice dispatched")
		}
	}
}

func TestBedrockClaudeRequiredChoiceNeverDowngrades(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; http.Error(w, "unsupported", 400) }))
	defer upstream.Close()
	p := &BedrockProvider{baseURL: upstream.URL, client: upstream.Client(), bearerToken: "test-only"}
	req := &types.ChatCompletionRequest{ToolChoice: json.RawMessage(`"required"`), Tools: []types.Tool{{Type: "function", Function: types.FunctionDef{Name: "read"}}}}
	if _, err := p.Send(context.Background(), req, "us.anthropic.claude-sonnet-5"); err == nil {
		t.Fatal("rejection hidden")
	}
	if _, err := p.SendStream(context.Background(), req, "us.anthropic.claude-sonnet-5"); err == nil {
		t.Fatal("stream rejection hidden")
	}
	if calls != 2 {
		t.Fatal("constraint retried")
	}
}
