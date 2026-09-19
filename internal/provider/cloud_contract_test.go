package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

type cloudLocalTransport struct {
	host string
	base http.RoundTripper
}

func (t cloudLocalTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Scheme != "http" || r.URL.Host != t.host {
		return nil, fmt.Errorf("offline contract test blocked destination %s", r.URL.Host)
	}
	return t.base.RoundTrip(r)
}

func cloudClient(server *httptest.Server) *http.Client {
	return &http.Client{Transport: cloudLocalTransport{
		host: strings.TrimPrefix(server.URL, "http://"), base: server.Client().Transport}}
}

func cloudRequest() *types.ChatCompletionRequest {
	limit := 100
	return &types.ChatCompletionRequest{
		Model: "aion-auto", MaxTokens: &limit,
		Messages: []types.Message{
			{Role: "system", Content: json.RawMessage(`"Synthetic contract test."`)},
			{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"Evidence","cache_control":{"type":"ephemeral","ttl":"5m"}}]`)},
			{Role: "assistant", ToolCalls: []types.ToolCall{{ID: "read-1", Type: "function", Function: types.FunctionCall{Name: "read", Arguments: `{"source":"fixture"}`}}}},
			{Role: "tool", ToolCallID: "read-1", Content: json.RawMessage(`"read-only result"`)},
			{Role: "user", Content: json.RawMessage(`"Submit the result."`)},
		},
		Tools: []types.Tool{
			{Type: "function", Function: types.FunctionDef{Name: "read", Parameters: json.RawMessage(`{"type":"object"}`)}},
			{Type: "function", Function: types.FunctionDef{Name: "submit_decision", Parameters: json.RawMessage(`{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}`)}},
		},
		ToolChoice:      json.RawMessage(`{"type":"function","function":{"name":"submit_decision"}}`),
		AIONPreferences: &types.AIONPreferences{SessionID: "local-private-session"},
	}
}

func TestCloudOfflineStreamingAndRefusal(t *testing.T) {
	for _, cloud := range []string{"azure-v1", "vertex-openai", "vertex-claude"} {
		t.Run(cloud, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				raw, _ := io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				if cloud == "vertex-claude" {
					if !strings.HasSuffix(r.URL.Path, ":streamRawPredict") ||
						!strings.Contains(string(raw), `"tool_choice":{"type":"tool","name":"submit_decision"}`) ||
						!strings.Contains(string(raw), `"cache_control"`) {
						t.Error("Vertex streaming contract lost")
					}
					io.WriteString(w, "event: message_start\ndata: "+`{"message":{"id":"blocked","model":"fixture","usage":{"input_tokens":7,"cache_read_input_tokens":4000}}}`+"\n\n")
					io.WriteString(w, "event: message_delta\ndata: "+`{"delta":{"stop_reason":"refusal"},"usage":{"output_tokens":0}}`+"\n\n")
					io.WriteString(w, "event: message_stop\ndata: {}\n\n")
				} else {
					if !strings.Contains(string(raw), `"stream":true`) ||
						!strings.Contains(string(raw), `"tool_choice":{"type":"function"`) ||
						strings.Contains(string(raw), `"cache_control"`) {
						t.Error("compatible streaming contract changed")
					}
					io.WriteString(w, "data: "+`{"id":"blocked","choices":[{"index":0,"delta":{"role":"assistant","refusal":"Fixture refusal."},"finish_reason":null}]}`+"\n\n")
					io.WriteString(w, "data: "+`{"id":"blocked","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":12,"completion_tokens":0,"total_tokens":12}}`+"\n\n")
					io.WriteString(w, "data: [DONE]\n\n")
				}
			}))
			defer server.Close()
			cfg := &config.ProviderConfig{APIKey: "local-test-only", BaseURL: server.URL, ProjectID: "test", Region: "us-central1"}
			var p Provider
			switch cloud {
			case "vertex-claude":
				x := mustVertex(t, cfg)
				x.client = cloudClient(server)
				p = x
			case "vertex-openai":
				x := mustGemini(t, cfg)
				x.client = cloudClient(server)
				p = x
			default:
				x := mustOpenAI(t, cfg)
				x.client = cloudClient(server)
				p = x
			}
			stream, err := p.SendStream(context.Background(), cloudRequest(), "fixture")
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			var finish, refusal string
			for {
				chunk, err := stream.ReadChunk()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				for _, choice := range chunk.Choices {
					if choice.FinishReason != nil {
						finish = *choice.FinishReason
					}
					if choice.Delta.Refusal != nil {
						refusal += *choice.Delta.Refusal
					}
				}
			}
			if calls != 1 || (cloud == "vertex-claude" && finish != "refusal") ||
				(cloud != "vertex-claude" && (finish != "content_filter" || refusal != "Fixture refusal.")) {
				t.Fatal("streaming refusal lost or retried")
			}
		})
	}
}

func TestCloudOfflineVertexRejectsUnsupportedContractsBeforeNetwork(t *testing.T) {
	requests := []*types.ChatCompletionRequest{
		{ToolChoice: json.RawMessage(`"required"`)},
		{ToolChoice: json.RawMessage(`{"type":"function","function":{"name":"missing"}}`)},
		{ResponseFormat: &types.ResponseFormat{Type: "json_schema"}},
		{SchemaSettings: &types.SchemaSettings{MustEmitNative: true}},
	}
	p := mustVertex(t, &config.ProviderConfig{})
	// A nil transport makes accidental dispatch fail the test immediately.
	p.client = nil
	for _, request := range requests {
		if _, err := p.Send(context.Background(), request, "fixture"); err == nil {
			t.Fatal("unsupported contract accepted")
		}
		if _, err := p.SendStream(context.Background(), request, "fixture"); err == nil {
			t.Fatal("unsupported streaming contract accepted")
		}
	}
}

func TestCloudOfflineVertexPreservesChoiceModesAndProviderErrors(t *testing.T) {
	for _, mode := range []struct{ input, want string }{
		{`"auto"`, `{"type":"auto"}`}, {`"none"`, `{"type":"none"}`}, {`"required"`, `{"type":"any"}`},
	} {
		t.Run(mode.input, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body map[string]json.RawMessage
				json.NewDecoder(r.Body).Decode(&body)
				if string(body["tool_choice"]) != mode.want {
					t.Error("tool choice weakened")
				}
				http.Error(w, "fixture unsupported model setting", http.StatusBadRequest)
			}))
			defer server.Close()
			p := mustVertex(t, &config.ProviderConfig{BaseURL: server.URL})
			p.client = cloudClient(server)
			req := cloudRequest()
			req.ToolChoice = json.RawMessage(mode.input)
			if _, err := p.Send(context.Background(), req, "fixture"); err == nil {
				t.Fatal("provider rejection hidden")
			}
			if _, err := p.SendStream(context.Background(), req, "fixture"); err == nil {
				t.Fatal("streaming provider rejection hidden")
			}
			if calls != 2 {
				t.Fatal("provider rejection retried")
			}
		})
	}
}

const cloudOpenAIResponse = `{"id":"local-contract","model":"fixture-deployment","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"decision-1","type":"function","function":{"name":"submit_decision","arguments":"{\"value\":9007199254740993}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4107,"completion_tokens":4,"total_tokens":4111,"prompt_tokens_details":{"cached_tokens":4000}}}`
const cloudClaudeResponse = `{"id":"local-contract","content":[{"type":"tool_use","id":"decision-1","name":"submit_decision","input":{"value":9007199254740993}}],"stop_reason":"tool_use","usage":{"input_tokens":7,"output_tokens":4,"cache_read_input_tokens":4000,"cache_creation_input_tokens":100}}`

func TestCloudOfflineRequestAndResponseContracts(t *testing.T) {
	for _, cloud := range []string{"azure-v1", "vertex-openai", "gemini", "vertex-claude"} {
		t.Run(cloud, func(t *testing.T) {
			calls := 0
			basePath := "/openai/v1"
			if cloud == "vertex-openai" {
				basePath = "/v1/projects/test-project/locations/us-central1/endpoints/openapi"
			} else if cloud == "gemini" {
				basePath = "/v1beta/openai"
			} else if cloud == "vertex-claude" {
				basePath = ""
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				raw, _ := io.ReadAll(r.Body)
				var body map[string]json.RawMessage
				if err := json.Unmarshal(raw, &body); err != nil {
					t.Error(err)
					return
				}
				if r.Method != http.MethodPost || r.Header.Get("Authorization") != "Bearer local-test-only" {
					t.Error("method or bearer credential mapping changed")
				}
				if body["aion_preferences"] != nil || body["SchemaSettings"] != nil {
					t.Error("internal controls leaked")
				}
				if cloud == "vertex-claude" {
					if r.URL.Path != "/v1/projects/test-project/locations/us-central1/publishers/anthropic/models/fixture-deployment:rawPredict" {
						t.Error("Vertex resource path changed")
					}
					if string(body["anthropic_version"]) != `"vertex-2023-10-16"` || body["model"] != nil {
						t.Error("Vertex protocol version or model placement changed")
					}
					if string(body["tool_choice"]) != `{"type":"tool","name":"submit_decision"}` {
						t.Error("Vertex dropped the forced named tool")
					}
					if strings.Count(string(raw), `"cache_control"`) != 1 {
						t.Error("Vertex dropped the native cache checkpoint")
					}
					if !strings.Contains(string(raw), `"tool_use_id":"read-1"`) || !strings.Contains(string(raw), "read-only result") {
						t.Error("Vertex lost the tool receipt")
					}
					io.WriteString(w, cloudClaudeResponse)
				} else {
					if r.URL.Path != basePath+"/chat/completions" || r.URL.RawQuery != "" {
						t.Error("OpenAI-compatible resource path changed")
					}
					if string(body["model"]) != `"fixture-deployment"` || string(body["tool_choice"]) != `{"type":"function","function":{"name":"submit_decision"}}` {
						t.Error("deployment or named tool changed")
					}
					if strings.Contains(string(raw), `"cache_control"`) {
						t.Error("Anthropic cache directive leaked to compatible endpoint")
					}
					if !strings.Contains(string(raw), `"tool_call_id":"read-1"`) {
						t.Error("compatible endpoint lost the tool receipt")
					}
					io.WriteString(w, cloudOpenAIResponse)
				}
			}))
			defer server.Close()
			cfg := &config.ProviderConfig{APIKey: "local-test-only", BaseURL: server.URL + basePath, ProjectID: "test-project", Region: "us-central1"}
			var p Provider
			switch cloud {
			case "vertex-claude":
				x := mustVertex(t, cfg)
				x.client = cloudClient(server)
				p = x
			case "gemini", "vertex-openai":
				x := mustGemini(t, cfg)
				x.client = cloudClient(server)
				p = x
			default:
				x := mustOpenAI(t, cfg)
				x.client = cloudClient(server)
				p = x
			}
			req := cloudRequest()
			before, _ := json.Marshal(req)
			result, err := p.Send(context.Background(), req, "fixture-deployment")
			if err != nil {
				t.Fatal(err)
			}
			after, _ := json.Marshal(req)
			choice := result.ChatResponse.Choices[0]
			if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) != 1 ||
				choice.Message.ToolCalls[0].Function.Arguments != `{"value":9007199254740993}` {
				t.Fatal("native structured output changed")
			}
			result.ChatResponse.Usage.NormalizeInputPartition()
			u := result.ChatResponse.Usage
			uncached, write := 107, 0
			if cloud == "vertex-claude" {
				uncached, write = 7, 100
			}
			if u.UncachedInputTokens != uncached || u.CacheReadInputTokens != 4000 || u.CacheCreationInputTokens != write || u.TotalTokens != 4111 {
				t.Fatalf("cache accounting changed: %+v", u)
			}
			if calls != 1 || string(before) != string(after) {
				t.Fatal("unexpected retry or caller mutation")
			}
		})
	}
}

func TestCloudOfflineRefusalsRemainVisible(t *testing.T) {
	for _, status := range []string{"stop", "content_filter"} {
		t.Run(status, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				fmt.Fprintf(w, `{"id":"blocked","choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"Fixture policy refusal."},"finish_reason":%q}],"usage":{"prompt_tokens":12,"completion_tokens":0,"total_tokens":12}}`, status)
			}))
			defer server.Close()
			p := mustOpenAI(t, &config.ProviderConfig{APIKey: "local-test-only", BaseURL: server.URL})
			p.client = cloudClient(server)
			result, err := p.Send(context.Background(), cloudRequest(), "fixture-deployment")
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := json.Marshal(result.ChatResponse)
			if !strings.Contains(string(raw), `"refusal":"Fixture policy refusal."`) ||
				result.ChatResponse.Choices[0].FinishReason != status || calls != 1 {
				t.Fatal("refusal hidden or retried")
			}
		})
	}
}

func TestCloudOfflineGuardRejectsExternalDestination(t *testing.T) {
	transport := cloudLocalTransport{host: "127.0.0.1:1"}
	req, _ := http.NewRequest(http.MethodPost, "https://example.invalid/inference", nil)
	if _, err := transport.RoundTrip(req); err == nil {
		t.Fatal("external destination was admitted")
	}
}

func mustOpenAI(t *testing.T, cfg *config.ProviderConfig) *OpenAIProvider {
	t.Helper()
	p, err := NewOpenAI(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func mustVertex(t *testing.T, cfg *config.ProviderConfig) *VertexProvider {
	t.Helper()
	p, err := NewVertex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func mustGemini(t *testing.T, cfg *config.ProviderConfig) *GeminiProvider {
	t.Helper()
	p, err := NewGemini(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
