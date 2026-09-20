package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

func TestBedrockMantleReasoningAndStreaming(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "json", true: "sse"}[stream], func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/chat/completions" || r.Header.Get("Authorization") != "Bearer local-test" {
					t.Errorf("incorrect Mantle path or authentication")
				}
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				for _, key := range []string{"anthropic_version", "max_tokens", "aion_preferences"} {
					if _, ok := body[key]; ok {
						t.Errorf("unexpected field %s", key)
					}
				}
				if string(body["reasoning_effort"]) != `"max"` || string(body["max_completion_tokens"]) != "128" {
					t.Errorf("missing reasoning or completion bound: %s", body)
				}
				if stream {
					if len(body["stream_options"]) == 0 {
						t.Error("usage not requested")
					}
					io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\ndata: [DONE]\n\n")
				} else {
					io.WriteString(w, `{"model":"openai.gpt-5.6-luna","choices":[{"message":{"role":"assistant","content":"OK"}}],"usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102,"prompt_tokens_details":{"cached_tokens":50,"cache_write_tokens":20}}}`)
				}
			}))
			defer server.Close()
			p, err := NewBedrock(&config.ProviderConfig{APIKey: "local-test", MantleBaseURL: server.URL, Models: []config.ModelConfig{{ID: "openai.gpt-5.6-luna", ReasoningEffort: "max"}}})
			if err != nil {
				t.Fatal(err)
			}
			limit := 128
			req := &types.ChatCompletionRequest{MaxTokens: &limit, Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"Hi"`)}}}
			if stream {
				r, err := p.SendStream(context.Background(), req, "openai.gpt-5.6-luna")
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				if _, err = r.ReadChunk(); err != nil {
					t.Fatal(err)
				}
				usage, err := r.ReadChunk()
				if err != nil || usage.Usage == nil || usage.Usage.CompletionTokens != 2 {
					t.Fatalf("missing usage: %#v %v", usage, err)
				}
				if _, err = r.ReadChunk(); err != io.EOF {
					t.Fatalf("want EOF: %v", err)
				}
			} else {
				r, err := p.Send(context.Background(), req, "openai.gpt-5.6-luna")
				if err != nil {
					t.Fatal(err)
				}
				u := r.ChatResponse.Usage
				if u.UncachedInputTokens != 30 || u.CacheReadInputTokens != 50 || u.CacheCreationInputTokens != 20 {
					t.Fatalf("wrong partition: %+v", u)
				}
			}
			if req.ReasoningEffort != "" || req.MaxTokens != &limit {
				t.Fatal("caller request mutated")
			}
		})
	}
}

func TestBedrockMantleInvalidReasoningFailsBeforeNetwork(t *testing.T) {
	for _, m := range []config.ModelConfig{{ID: "openai.gpt-5.6-luna", ReasoningEffort: "maximum"}, {ID: "anthropic.claude", ReasoningEffort: "max"}} {
		if _, err := NewBedrock(&config.ProviderConfig{APIKey: "test", Models: []config.ModelConfig{m}}); err == nil {
			t.Fatal("invalid model default accepted")
		}
	}
	p, err := NewBedrock(&config.ProviderConfig{APIKey: "test", MantleBaseURL: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = p.Send(context.Background(), &types.ChatCompletionRequest{ReasoningEffort: "maximum"}, "openai.gpt-5.6-luna"); err == nil || err.Error() != "bedrock mantle: invalid reasoning_effort" {
		t.Fatalf("wrong failure: %v", err)
	}
}

func TestBedrockMantleProviderErrorIsBoundedAndNotSilentlyRetried(t *testing.T) {
	for _, status := range []int{401, 403, 429, 503} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				w.WriteHeader(status)
				io.WriteString(w, "private prompt echoed by provider")
			}))
			defer server.Close()
			p, err := NewBedrock(&config.ProviderConfig{APIKey: "test", MantleBaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			_, err = p.Send(context.Background(), &types.ChatCompletionRequest{}, "openai.gpt-5.6-luna")
			if err == nil || err.Error() != fmt.Sprintf("bedrock mantle: HTTP %d", status) || calls != 1 {
				t.Fatalf("unexpected error handling: %v, calls=%d", err, calls)
			}
		})
	}
}

func TestBedrockMantleCancellationAndDeadline(t *testing.T) {
	for _, timeout := range []bool{false, true} {
		t.Run(fmt.Sprint(timeout), func(t *testing.T) {
			var calls atomic.Int32
			started := make(chan struct{})
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.Copy(io.Discard, r.Body)
				calls.Add(1)
				close(started)
				select {
				case <-r.Context().Done():
				case <-release:
				}
			}))
			defer server.Close()
			defer close(release)
			p, err := NewBedrock(&config.ProviderConfig{APIKey: "test", MantleBaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			if timeout {
				ctx, cancel = context.WithTimeout(context.Background(), 100*time.Millisecond)
			}
			defer cancel()
			result := make(chan error, 1)
			go func() { _, err := p.Send(ctx, &types.ChatCompletionRequest{}, "openai.gpt-5.6-luna"); result <- err }()
			select {
			case <-started:
			case <-time.After(2 * time.Second):
				t.Fatal("request not started")
			}
			if !timeout {
				cancel()
			}
			select {
			case err := <-result:
				want := context.Canceled
				if timeout {
					want = context.DeadlineExceeded
				}
				if !errors.Is(err, want) {
					t.Fatalf("want %v, got %v", want, err)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("cancelled request did not return")
			}
			if calls.Load() != 1 {
				t.Fatal("request retried")
			}
		})
	}
}
