package provider

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
)

func offlineProvider(t *testing.T, name string, server *httptest.Server) Provider {
	t.Helper()
	cfg := &config.ProviderConfig{APIKey: "synthetic-only", BaseURL: server.URL,
		ProjectID: "fixture", Region: "us-central1"}
	switch name {
	case "vertex":
		p := mustVertex(t, cfg)
		p.client = cloudClient(server)
		return p
	case "gemini":
		p := mustGemini(t, cfg)
		p.client = cloudClient(server)
		return p
	default:
		p := mustOpenAI(t, cfg)
		p.client = cloudClient(server)
		return p
	}
}

func TestCloudOfflineHTTPFailuresAreTerminal(t *testing.T) {
	for _, name := range []string{"openai", "gemini", "vertex"} {
		for _, status := range []int{401, 403, 429, 500, 503} {
			t.Run(fmt.Sprintf("%s/%d", name, status), func(t *testing.T) {
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					calls.Add(1)
					w.Header().Set("Retry-After", "60")
					http.Error(w, `{"error":"synthetic cloud rejection"}`, status)
				}))
				defer server.Close()
				p := offlineProvider(t, name, server)
				if _, err := p.Send(t.Context(), cloudRequest(), "fixture"); err == nil {
					t.Fatal("normal rejection hidden")
				}
				if _, err := p.SendStream(t.Context(), cloudRequest(), "fixture"); err == nil {
					t.Fatal("streaming rejection hidden")
				}
				if calls.Load() != 2 {
					t.Fatalf("inference retried: %d calls", calls.Load())
				}
			})
		}
	}
}

func TestCloudOfflineDeadlineStopsDispatch(t *testing.T) {
	for _, name := range []string{"openai", "gemini", "vertex"} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", name, streaming), func(t *testing.T) {
				var calls atomic.Int32
				release := make(chan struct{})
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					<-release
				}))
				defer server.Close()
				defer close(release)
				p := offlineProvider(t, name, server)
				ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
				defer cancel()
				var err error
				if streaming {
					_, err = p.SendStream(ctx, cloudRequest(), "fixture")
				} else {
					_, err = p.Send(ctx, cloudRequest(), "fixture")
				}
				if !errors.Is(err, context.DeadlineExceeded) || calls.Load() != 1 {
					t.Fatalf("deadline or dispatch count changed: %v, %d", err, calls.Load())
				}
			})
		}
	}
}

func TestCloudOfflineInterruptedStreamsAreErrors(t *testing.T) {
	for _, name := range []string{"openai", "gemini", "vertex"} {
		for _, failure := range []string{"disconnect", "provider-error", "malformed"} {
			t.Run(name+"/"+failure, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					if name == "vertex" {
						io.WriteString(w, "event: message_start\ndata: {\"message\":{\"id\":\"fixture\",\"usage\":{\"input_tokens\":10}}}\n\n")
						io.WriteString(w, "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n")
						if failure == "provider-error" {
							io.WriteString(w, "event: error\ndata: {\"error\":{\"message\":\"private-fixture-detail\"}}\n\nevent: message_stop\ndata: {}\n\n")
						} else if failure == "malformed" {
							io.WriteString(w, "event: content_block_delta\ndata: {broken}\n\n")
						}
					} else {
						io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"12\"},\"finish_reason\":\"stop\"}]}\n\n")
						if failure == "provider-error" {
							io.WriteString(w, "data: {\"error\":{\"message\":\"private-fixture-detail\"}}\n\ndata: [DONE]\n\n")
						} else if failure == "malformed" {
							io.WriteString(w, "data: {broken}\n\n")
						}
					}
				}))
				defer server.Close()
				p := offlineProvider(t, name, server)
				stream, err := p.SendStream(t.Context(), cloudRequest(), "fixture")
				if err != nil {
					t.Fatal(err)
				}
				defer stream.Close()
				for chunks := 0; chunks < 10; chunks++ {
					_, err = stream.ReadChunk()
					if err != nil {
						break
					}
				}
				if err == nil || err == io.EOF || strings.Contains(err.Error(), "private-fixture-detail") {
					t.Fatalf("interruption hidden or provider detail leaked: %v", err)
				}
			})
		}
	}
}
