package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/pricing"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/types"
)

// stubProvider returns a fixed non-stream response.
type stubProvider struct{ content string }

func (stubProvider) Name() string { return "bedrock" }
func (p stubProvider) Send(context.Context, *types.ChatCompletionRequest, string) (*provider.Response, error) {
	return &provider.Response{
		StatusCode: http.StatusOK,
		ChatResponse: &types.ChatCompletionResponse{
			Choices: []types.Choice{{
				Message:      types.Message{Role: "assistant", Content: json.RawMessage(`"` + p.content + `"`)},
				FinishReason: "stop",
			}},
			Usage: types.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		},
	}, nil
}
func (stubProvider) SendStream(context.Context, *types.ChatCompletionRequest, string) (provider.StreamReader, error) {
	return nil, nil
}

func stubHandler(content string) *Handler {
	cfg := &config.Config{}
	cfg.Providers.Bedrock = &config.ProviderConfig{
		Models: []config.ModelConfig{{ID: "haiku", Tier: 1, InputPricePer1M: 1, OutputPricePer1M: 2}},
	}
	reg := provider.NewRegistry()
	reg.Register(stubProvider{content: content})
	return &Handler{router: router.NewRouter(cfg, nil), registry: reg, pricing: pricing.NewTable(cfg.Providers)}
}

// A slow PostResponse hook must NOT delay the customer response. Run a real
// httptest.Server (ResponseRecorder cannot expose net/http buffering): the
// client must read the full body while the hook is still blocked. A synchronous
// hook would hold the body until handler return and this test would deadlock on
// the read.
func TestChatCompletion_PostResponseDoesNotDelayResponse(t *testing.T) {
	h := stubHandler("hello")
	release := make(chan struct{})
	hookEntered := make(chan struct{})
	hookSawContent := make(chan []string, 1)
	h.SetGatewayHooks(&types.GatewayHooks{
		PostResponse: func(in types.PostResponseInput) {
			close(hookEntered)
			hookSawContent <- in.ResponseContents
			<-release // block until the client has already read the body
		},
	})

	srv := httptest.NewServer(http.HandlerFunc(h.ChatCompletion))
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"haiku","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	// Read the full body with a deadline. If the hook were synchronous this read
	// would not complete until release is closed, so a timeout here is the failure.
	bodyCh := make(chan []byte, 1)
	go func() {
		b, _ := io.ReadAll(resp.Body)
		bodyCh <- b
	}()

	var body []byte
	select {
	case body = <-bodyCh:
	case <-time.After(2 * time.Second):
		close(release)
		t.Fatal("response body was delayed by the PostResponse hook (still blocked after 2s)")
	}
	if len(body) == 0 {
		close(release)
		t.Fatal("response body must be served")
	}

	// The hook still runs (asynchronously) and sees all choice contents.
	select {
	case <-hookEntered:
	case <-time.After(2 * time.Second):
		close(release)
		t.Fatal("PostResponse hook never ran")
	}
	got := <-hookSawContent
	if len(got) != 1 || got[0] != "hello" {
		t.Errorf("hook must see all parsed choice contents, got %v", got)
	}
	close(release)
}
