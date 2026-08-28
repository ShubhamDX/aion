package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
	"github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/credentials"
)

func TestBedrockAuthorizeBearer(t *testing.T) {
	provider := &BedrockProvider{bearerToken: "short-lived-token"}
	request, err := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := provider.authorize(t.Context(), request, []byte(`{"max_tokens":1}`)); err != nil {
		t.Fatal(err)
	}
	if got := request.Header.Get("Authorization"); got != "Bearer short-lived-token" {
		t.Fatalf("authorization = %q", got)
	}
}

func TestBedrockAuthorizeSigV4(t *testing.T) {
	provider := &BedrockProvider{
		credentials: credentials.NewStaticCredentialsProvider("access-key", "secret-key", "session-token"),
		signer:      v4.NewSigner(),
		region:      "us-east-1",
	}
	request, err := http.NewRequestWithContext(context.Background(), http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := provider.authorize(t.Context(), request, []byte(`{"max_tokens":1}`)); err != nil {
		t.Fatal(err)
	}
	if got := request.Header.Get("Authorization"); !strings.HasPrefix(got, "AWS4-HMAC-SHA256 ") {
		t.Fatalf("authorization = %q", got)
	}
	if request.Header.Get("X-Amz-Security-Token") != "session-token" {
		t.Fatal("session token was not included in the signed request")
	}
}

// The model is conveyed via the invoke URL path, not the body: Bedrock
// resolves the model from the URL, and DeepSeek-R1 rejects a request body
// containing an unrecognized "model" field with a 400 validation error.
func TestBedrockRequestBodyOmitsModelField(t *testing.T) {
	provider := &BedrockProvider{}
	req := &types.ChatCompletionRequest{
		Model:    "aion-escalate",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}

	bReq := provider.translateRequest(req, false)

	body, err := json.Marshal(bReq)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatal(err)
	}
	if _, ok := decoded["model"]; ok {
		t.Fatalf("bedrock request body must not carry a model field: %s", body)
	}
}
