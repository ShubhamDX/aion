package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"cloud.google.com/go/auth"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/ShubhamDX/aion/internal/config"
	pkgtypes "github.com/ShubhamDX/aion/pkg/types"
)

type tokenFunc func(context.Context) (*auth.Token, error)

func (f tokenFunc) Token(ctx context.Context) (*auth.Token, error) { return f(ctx) }

type cloudRoundTripFunc func(*http.Request) (*http.Response, error)

func (f cloudRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func cloudResponse(req *http.Request, status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(body)),
		Header: http.Header{"Content-Type": {"application/json"}}, Request: req}
}

func TestCloudCompatibleStreamingRequestsUsage(t *testing.T) {
	for _, name := range []string{"openai", "gemini"} {
		t.Run(name, func(t *testing.T) {
			var instance Provider
			var client *http.Client
			cfg := &config.ProviderConfig{BaseURL: "http://fixture.invalid"}
			if name == "openai" {
				p := mustOpenAI(t, cfg)
				instance, client = p, p.client
			} else {
				p := mustGemini(t, cfg)
				instance, client = p, p.client
			}
			client.Transport = cloudRoundTripFunc(func(req *http.Request) (*http.Response, error) {
				var payload struct {
					StreamOptions *pkgtypes.StreamOptions `json:"stream_options"`
				}
				if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
					t.Fatal(err)
				}
				if payload.StreamOptions == nil || !payload.StreamOptions.IncludeUsage {
					t.Fatal("explicit usage request lost")
				}
				return cloudResponse(req, 200, "data: [DONE]\n\n"), nil
			})
			request := cloudRequest()
			request.StreamOptions = &pkgtypes.StreamOptions{IncludeUsage: true}
			stream, err := instance.SendStream(context.Background(), request, "fixture")
			if err != nil {
				t.Fatal(err)
			}
			stream.Close()
		})
	}
}

func TestCloudAuthGoogleSDKRefreshAndConcurrentReuse(t *testing.T) {
	var refreshes atomic.Int32
	source := tokenFunc(func(ctx context.Context) (*auth.Token, error) {
		n := refreshes.Add(1)
		expiry := time.Now().Add(time.Hour)
		if n == 1 {
			expiry = time.Now().Add(time.Minute)
		}
		return &auth.Token{Value: fmt.Sprintf("fixture-token-%d", n), Expiry: expiry}, ctx.Err()
	})
	creds := auth.NewCredentials(&auth.CredentialsOptions{
		TokenProvider: auth.NewCachedTokenProvider(source, &auth.CachedTokenProviderOptions{
			ExpireEarly: 2 * time.Minute, DisableAsyncRefresh: true,
		}),
		QuotaProjectIDProvider: auth.CredentialsPropertyFunc(func(context.Context) (string, error) {
			return "fixture-quota-project", nil
		}),
	})
	origin, _ := url.Parse("https://us-central1-aiplatform.googleapis.com")
	transport := &cloudAuthTransport{origin: origin, bearer: googleBearer(creds),
		base: cloudRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.Header.Get("X-Goog-User-Project") != "fixture-quota-project" {
				t.Error("quota project missing")
			}
			return cloudResponse(req, 200, req.Header.Get("Authorization")), nil
		})}
	send := func(want string) {
		req, _ := http.NewRequest(http.MethodPost, origin.String()+"/v1/fixture", nil)
		req.Header.Set("Authorization", "Bearer caller-owned-value")
		resp, err := transport.RoundTrip(req)
		if err != nil {
			t.Error(err)
			return
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		if string(body) != want || req.Header.Get("Authorization") != "Bearer caller-owned-value" {
			t.Error("token selection or caller immutability failed")
		}
	}
	send("Bearer fixture-token-1")
	var group sync.WaitGroup
	for range 32 {
		group.Add(1)
		go func() { defer group.Done(); send("Bearer fixture-token-2") }()
	}
	group.Wait()
	if refreshes.Load() != 2 {
		t.Fatalf("unexpected refresh count: %d", refreshes.Load())
	}
}

func TestCloudAuthFailedRefreshDoesNotDispatchOrLeak(t *testing.T) {
	var refreshes, dispatches atomic.Int32
	creds := auth.NewCredentials(&auth.CredentialsOptions{
		TokenProvider: auth.NewCachedTokenProvider(tokenFunc(func(context.Context) (*auth.Token, error) {
			if refreshes.Add(1) == 1 {
				return &auth.Token{Value: "fixture-old-token", Expiry: time.Now().Add(time.Minute)}, nil
			}
			return nil, errors.New("identity body includes fixture-secret")
		}), &auth.CachedTokenProviderOptions{ExpireEarly: 2 * time.Minute, DisableAsyncRefresh: true}),
	})
	origin, _ := url.Parse("https://us-east5-aiplatform.googleapis.com")
	transport := &cloudAuthTransport{origin: origin, bearer: googleBearer(creds),
		base: cloudRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			dispatches.Add(1)
			return cloudResponse(req, 200, "{}"), nil
		})}
	req, _ := http.NewRequest(http.MethodPost, origin.String()+"/v1/fixture", nil)
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if _, err = transport.RoundTrip(req); !errors.Is(err, errCloudCredential) || strings.Contains(err.Error(), "fixture-secret") {
		t.Fatalf("credential failure not sanitized: %v", err)
	}
	if dispatches.Load() != 1 {
		t.Fatal("failed refresh dispatched or retried inference")
	}
}

func TestCloudAuthRejectsExpiredTokenCancellationAndOtherOrigin(t *testing.T) {
	for _, scenario := range []string{"expired", "cancelled", "origin"} {
		t.Run(scenario, func(t *testing.T) {
			origin, _ := url.Parse("https://us-east5-aiplatform.googleapis.com")
			creds := auth.NewCredentials(&auth.CredentialsOptions{TokenProvider: tokenFunc(func(context.Context) (*auth.Token, error) {
				return &auth.Token{Value: "expired-fixture", Expiry: time.Now().Add(-time.Minute)}, nil
			})})
			transport := &cloudAuthTransport{origin: origin, bearer: googleBearer(creds),
				base: cloudRoundTripFunc(func(*http.Request) (*http.Response, error) {
					t.Fatal("invalid authentication dispatched inference")
					return nil, nil
				})}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if scenario == "cancelled" {
				cancel()
			}
			endpoint := origin.String()
			if scenario == "origin" {
				endpoint = "https://other.invalid"
			}
			req, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
			if _, err := transport.RoundTrip(req); err == nil {
				t.Fatal("invalid authentication accepted")
			}
		})
	}
}

func TestCloudAuthAzureSDKRefreshAndReuse(t *testing.T) {
	const tenant = "00000000-0000-0000-0000-000000000001"
	const endpoint = "https://login.microsoftonline.com/" + tenant + "/oauth2/v2.0/token"
	var refreshes atomic.Int32
	client := &http.Client{Transport: cloudRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		if strings.HasSuffix(req.URL.Path, "/.well-known/openid-configuration") {
			return cloudResponse(req, 200, `{"authorization_endpoint":"https://login.microsoftonline.com/`+tenant+
				`/oauth2/v2.0/authorize","token_endpoint":"`+endpoint+`","issuer":"https://login.microsoftonline.com/`+tenant+`/v2.0"}`), nil
		}
		if req.URL.String() != endpoint || req.Method != http.MethodPost {
			t.Fatalf("unexpected identity request: %s %s", req.Method, req.URL)
		}
		if err := req.ParseForm(); err != nil {
			t.Fatal(err)
		}
		scopes := strings.Fields(req.Form.Get("scope"))
		found := false
		for _, scope := range scopes {
			switch scope {
			case azureInferenceScope:
				found = true
			case "openid", "offline_access", "profile":
			default:
				t.Fatalf("unexpected Azure scope: %q", scope)
			}
		}
		if !found {
			t.Fatalf("missing Azure inference scope: %q", req.Form.Get("scope"))
		}
		n := refreshes.Add(1)
		expiry := 3600
		if n == 1 {
			expiry = 60
		}
		return cloudResponse(req, 200, fmt.Sprintf(
			`{"token_type":"Bearer","expires_in":%d,"access_token":"azure-fixture-%d"}`, expiry, n)), nil
	})}
	credential, err := azidentity.NewClientSecretCredential(tenant, "00000000-0000-0000-0000-000000000002",
		"synthetic-secret", &azidentity.ClientSecretCredentialOptions{
			DisableInstanceDiscovery: true,
			ClientOptions:            azcore.ClientOptions{Transport: client, Retry: policy.RetryOptions{MaxRetries: -1}},
		})
	if err != nil {
		t.Fatal(err)
	}
	bearer := azureBearer(credential)
	for i, want := range []string{"azure-fixture-1", "azure-fixture-2", "azure-fixture-2"} {
		token, quota, err := bearer(context.Background())
		if err != nil || token != want || quota != "" {
			t.Fatalf("token %d: token=%q quota=%q error=%v", i, token, quota, err)
		}
	}
	if refreshes.Load() != 2 {
		t.Fatalf("unexpected Azure refresh count: %d", refreshes.Load())
	}
}

func TestCloudAuthConstructorsCoverNormalStreamingAndTerminalErrors(t *testing.T) {
	adcPath := filepath.Join(t.TempDir(), "adc.json")
	if err := os.WriteFile(adcPath, []byte(`{"type":"authorized_user","client_id":"fixture","client_secret":"fixture","refresh_token":"fixture"}`), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", adcPath)
	t.Setenv("AZURE_TOKEN_CREDENTIALS", "EnvironmentCredential")
	t.Setenv("AZURE_TENANT_ID", "00000000-0000-0000-0000-000000000001")
	t.Setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000002")
	t.Setenv("AZURE_CLIENT_SECRET", "synthetic-only")
	for _, name := range []string{"openai", "gemini", "vertex"} {
		t.Run(name, func(t *testing.T) {
			cfg := &config.ProviderConfig{ProjectID: "fixture", Region: "us-central1", CredentialMode: "google_adc"}
			var instance Provider
			var client *http.Client
			switch name {
			case "openai":
				cfg.CredentialMode, cfg.BaseURL = "azure_entra", "https://fixture.openai.azure.com/openai/v1"
				p := mustOpenAI(t, cfg)
				instance, client = p, p.client
			case "gemini":
				cfg.BaseURL = "https://us-central1-aiplatform.googleapis.com/v1/projects/fixture/locations/us-central1/endpoints/openapi"
				p := mustGemini(t, cfg)
				instance, client = p, p.client
			case "vertex":
				p := mustVertex(t, cfg)
				instance, client = p, p.client
			}
			transport, ok := client.Transport.(*cloudAuthTransport)
			if !ok || client.CheckRedirect == nil {
				t.Fatal("constructor omitted managed authentication or redirect restriction")
			}
			var tokenCalls, requests int
			transport.bearer = func(context.Context) (string, string, error) {
				tokenCalls++
				return "managed-fixture", "", nil
			}
			transport.base = cloudRoundTripFunc(func(req *http.Request) (*http.Response, error) {
				requests++
				if req.Header.Get("Authorization") != "Bearer managed-fixture" {
					t.Error("managed credential omitted")
				}
				// A rejection must not trigger credential fallback or retry.
				return cloudResponse(req, 401, `{"error":"fixture-denied"}`), nil
			})
			if _, err := instance.Send(context.Background(), cloudRequest(), "fixture"); err == nil {
				t.Fatal("normal rejection hidden")
			}
			if _, err := instance.SendStream(context.Background(), cloudRequest(), "fixture"); err == nil {
				t.Fatal("streaming rejection hidden")
			}
			if requests != 2 || tokenCalls != 2 {
				t.Fatal("inference rejection retried")
			}
			if err := client.CheckRedirect(&http.Request{}, nil); err == nil {
				t.Fatal("credentialed redirect accepted")
			}
		})
	}
}
