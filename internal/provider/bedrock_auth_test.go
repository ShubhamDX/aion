package provider

import (
	"context"
	"net/http"
	"strings"
	"testing"

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
