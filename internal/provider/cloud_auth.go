package provider

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"time"

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/ShubhamDX/aion/internal/config"
	pkgconfig "github.com/ShubhamDX/aion/pkg/config"
)

const googleCloudScope = "https://www.googleapis.com/auth/cloud-platform"
const azureInferenceScope = "https://ai.azure.com/.default"

var errCloudCredential = errors.New("cloud credential unavailable")

type cloudBearer func(context.Context) (token, quotaProject string, err error)

// cloudAuthTransport adds a credential only to the configured origin. Token
// caching and refresh belong to the official identity libraries.
type cloudAuthTransport struct {
	origin *url.URL
	bearer cloudBearer
	base   http.RoundTripper
}

func (t *cloudAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != t.origin.Scheme || req.URL.Host != t.origin.Host {
		return nil, errors.New("cloud credential destination mismatch")
	}
	if err := req.Context().Err(); err != nil {
		return nil, err
	}
	token, quota, err := t.bearer(req.Context())
	if err != nil || token == "" {
		if ctxErr := req.Context().Err(); ctxErr != nil {
			return nil, ctxErr
		}
		// Identity errors can contain credential response bodies. Do not expose
		// those through provider errors, telemetry or dashboard diagnostics.
		return nil, errCloudCredential
	}
	if err := req.Context().Err(); err != nil {
		return nil, err
	}
	clone := req.Clone(req.Context())
	clone.Header.Set("Authorization", "Bearer "+token)
	if quota != "" {
		clone.Header.Set("X-Goog-User-Project", quota)
	}
	return t.base.RoundTrip(clone)
}

func googleBearer(creds *auth.Credentials) cloudBearer {
	return func(ctx context.Context) (string, string, error) {
		token, err := creds.Token(ctx)
		if err != nil || token == nil || token.Value == "" || (!token.Expiry.IsZero() && !time.Now().Before(token.Expiry)) {
			return "", "", errCloudCredential
		}
		quota, err := creds.QuotaProjectID(ctx)
		if err != nil {
			return "", "", errCloudCredential
		}
		return token.Value, quota, nil
	}
}

func azureBearer(creds azcore.TokenCredential) cloudBearer {
	return func(ctx context.Context) (string, string, error) {
		token, err := creds.GetToken(ctx, policy.TokenRequestOptions{Scopes: []string{azureInferenceScope}})
		if err != nil || token.Token == "" || !time.Now().Before(token.ExpiresOn) {
			return "", "", errCloudCredential
		}
		return token.Token, "", nil
	}
}

func cloudHTTPClient(name string, cfg *config.ProviderConfig, baseURL string) (*http.Client, error) {
	if err := pkgconfig.ValidateCloudCredentials(name, cfg); err != nil {
		return nil, err
	}
	var bearer cloudBearer
	switch cfg.CredentialMode {
	case "google_adc":
		creds, err := credentials.DetectDefault(&credentials.DetectOptions{
			Scopes: []string{googleCloudScope}, DisableAsyncRefresh: true,
			Client: &http.Client{Timeout: 15 * time.Second},
		})
		if err != nil {
			return nil, errCloudCredential
		}
		bearer = googleBearer(creds)
	case "azure_entra":
		creds, err := azidentity.NewDefaultAzureCredential(nil)
		if err != nil {
			return nil, errCloudCredential
		}
		bearer = azureBearer(creds)
	default:
		return &http.Client{}, nil
	}
	origin, err := url.Parse(baseURL)
	if err != nil {
		return nil, errors.New("invalid cloud endpoint")
	}
	return &http.Client{
		Transport: &cloudAuthTransport{origin: origin, bearer: bearer, base: http.DefaultTransport},
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return errors.New("cloud credential redirects are disabled")
		},
	}, nil
}
