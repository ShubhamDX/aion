package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"cloud.google.com/go/auth/credentials"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
)

// Only the identity HTTP transport is synthetic. SDK parsing and refresh run
// unchanged, without credential discovery or an external connection.
func offlineIdentity(t *testing.T, family string, tokenHTTP cloudRoundTripFunc) cloudBearer {
	t.Helper()
	const tenant = "00000000-0000-0000-0000-000000000001"
	const azureToken = "https://login.microsoftonline.com/" + tenant + "/oauth2/v2.0/token"
	client := &http.Client{Transport: cloudRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		if family == "azure" && strings.HasSuffix(req.URL.Path, "/.well-known/openid-configuration") {
			return cloudResponse(req, 200, `{"authorization_endpoint":"https://login.microsoftonline.com/`+tenant+
				`/oauth2/v2.0/authorize","token_endpoint":"`+azureToken+`","issuer":"https://login.microsoftonline.com/`+tenant+`/v2.0"}`), nil
		}
		want := "https://oauth2.googleapis.com/token"
		grant := "refresh_token"
		if family == "azure" {
			want, grant = azureToken, "client_credentials"
		}
		if req.Method != http.MethodPost || req.URL.String() != want {
			return nil, fmt.Errorf("offline identity blocked destination")
		}
		if err := req.ParseForm(); err != nil || req.Form.Get("grant_type") != grant {
			return nil, fmt.Errorf("unexpected identity grant")
		}
		return tokenHTTP(req)
	})}
	if family == "google" {
		creds, err := credentials.DetectDefault(&credentials.DetectOptions{
			CredentialsJSON: []byte(`{"type":"authorized_user","client_id":"synthetic","client_secret":"synthetic","refresh_token":"synthetic"}`),
			Scopes:          []string{googleCloudScope}, EarlyTokenRefresh: 2 * time.Minute,
			DisableAsyncRefresh: true, Client: client,
		})
		if err != nil {
			t.Fatal(err)
		}
		return googleBearer(creds)
	}
	creds, err := azidentity.NewClientSecretCredential(tenant,
		"00000000-0000-0000-0000-000000000002", "synthetic", &azidentity.ClientSecretCredentialOptions{
			DisableInstanceDiscovery: true,
			ClientOptions:            azcore.ClientOptions{Transport: client, Retry: policy.RetryOptions{MaxRetries: -1}},
		})
	if err != nil {
		t.Fatal(err)
	}
	return azureBearer(creds)
}

func TestCloudOfflineIdentityRefreshConcurrency(t *testing.T) {
	for _, family := range []string{"google", "azure"} {
		t.Run(family, func(t *testing.T) {
			var refreshes atomic.Int32
			bearer := offlineIdentity(t, family, func(req *http.Request) (*http.Response, error) {
				n := refreshes.Add(1)
				expiry := 3600
				if n == 1 {
					expiry = -1
				}
				return cloudResponse(req, 200, fmt.Sprintf(
					`{"token_type":"Bearer","expires_in":%d,"access_token":"fixture-%d"}`, expiry, n)), nil
			})
			token, _, err := bearer(t.Context())
			if !errors.Is(err, errCloudCredential) || token != "" {
				t.Fatalf("initial token: %q, %v", token, err)
			}
			var group sync.WaitGroup
			for range 32 {
				group.Add(1)
				go func() {
					defer group.Done()
					token, _, err := bearer(t.Context())
					if err != nil || token != "fixture-2" {
						t.Errorf("refreshed token: %q, %v", token, err)
					}
				}()
			}
			group.Wait()
			if refreshes.Load() != 2 {
				t.Fatalf("refresh stampede: %d exchanges", refreshes.Load())
			}
		})
	}
}

func TestCloudOfflineIdentityFailureAndCancellation(t *testing.T) {
	for _, family := range []string{"google", "azure"} {
		for _, scenario := range []string{"refresh-rejected", "cancelled"} {
			t.Run(family+"/"+scenario, func(t *testing.T) {
				var calls, dispatched atomic.Int32
				entered := make(chan struct{}, 1)
				bearer := offlineIdentity(t, family, func(req *http.Request) (*http.Response, error) {
					if calls.Add(1) == 1 && scenario == "refresh-rejected" {
						return cloudResponse(req, 200, `{"token_type":"Bearer","expires_in":-1,"access_token":"fixture-old"}`), nil
					}
					if scenario == "cancelled" {
						entered <- struct{}{}
						<-req.Context().Done()
						return nil, req.Context().Err()
					}
					return cloudResponse(req, 400, `{"error":"invalid_grant","error_description":"private-fixture-detail"}`), nil
				})
				if scenario == "refresh-rejected" {
					if _, _, err := bearer(t.Context()); !errors.Is(err, errCloudCredential) {
						t.Fatalf("expired initial token accepted: %v", err)
					}
				}
				origin, _ := url.Parse("https://cloud-fixture.invalid")
				transport := &cloudAuthTransport{origin: origin, bearer: bearer,
					base: cloudRoundTripFunc(func(*http.Request) (*http.Response, error) {
						dispatched.Add(1)
						return nil, errors.New("unexpected inference")
					})}
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				req, _ := http.NewRequestWithContext(ctx, http.MethodPost, origin.String(), nil)
				result := make(chan error, 1)
				go func() { _, err := transport.RoundTrip(req); result <- err }()
				if scenario == "cancelled" {
					select {
					case <-entered:
						cancel()
					case <-time.After(2 * time.Second):
						t.Fatal("token request not reached")
					}
				}
				select {
				case err := <-result:
					want := errCloudCredential
					if scenario == "cancelled" {
						want = context.Canceled
					}
					if !errors.Is(err, want) || strings.Contains(err.Error(), "private-fixture-detail") || dispatched.Load() != 0 {
						t.Fatalf("identity failure leaked or dispatched: %v", err)
					}
				case <-time.After(2 * time.Second):
					t.Fatal("identity failure did not finish")
				}
			})
		}
	}
}
