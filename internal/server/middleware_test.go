package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/config"
)

func TestAuthMiddleware_DefaultKeepsExtraRoutesProtected(t *testing.T) {
	handler := protectedHandler(AuthMiddleware(testValidator()))

	res := httptest.NewRecorder()
	handler.ServeHTTP(res, httptest.NewRequest(http.MethodGet, "/dashboard", nil))

	if res.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", res.Code, http.StatusUnauthorized)
	}
}

func TestAuthMiddleware_ExtraBypassPrefix(t *testing.T) {
	handler := protectedHandler(AuthMiddlewareWithOptions(testValidator(), AuthOptions{
		ExtraBypassPrefixes: []string{"/dashboard"},
	}))

	for _, path := range []string{"/dashboard", "/dashboard/", "/dashboard/api/overview"} {
		t.Run(path, func(t *testing.T) {
			res := httptest.NewRecorder()
			handler.ServeHTTP(res, httptest.NewRequest(http.MethodGet, path, nil))
			if res.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", res.Code, http.StatusOK)
			}
		})
	}
}

func TestAuthMiddleware_ExtraBypassPrefixDoesNotMatchSibling(t *testing.T) {
	handler := protectedHandler(AuthMiddlewareWithOptions(testValidator(), AuthOptions{
		ExtraBypassPrefixes: []string{"/dashboard"},
	}))

	res := httptest.NewRecorder()
	handler.ServeHTTP(res, httptest.NewRequest(http.MethodGet, "/dashboard2", nil))

	if res.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", res.Code, http.StatusUnauthorized)
	}
}

func TestAuthMiddleware_HealthBypassPreserved(t *testing.T) {
	handler := protectedHandler(AuthMiddlewareWithOptions(testValidator(), AuthOptions{}))

	for _, path := range []string{"/health", "/healthz/license"} {
		t.Run(path, func(t *testing.T) {
			res := httptest.NewRecorder()
			handler.ServeHTTP(res, httptest.NewRequest(http.MethodGet, path, nil))
			if res.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", res.Code, http.StatusOK)
			}
		})
	}
}

func TestAuthMiddleware_ValidAPIKeyStillWorks(t *testing.T) {
	handler := protectedHandler(AuthMiddlewareWithOptions(testValidator(), AuthOptions{
		ExtraBypassPrefixes: []string{"/dashboard"},
	}))
	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer test-key")

	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", res.Code, http.StatusOK)
	}
}

func protectedHandler(middleware func(http.Handler) http.Handler) http.Handler {
	return middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
}

func testValidator() *apikey.Validator {
	return apikey.NewValidator([]config.KeyConfig{{Key: "test-key", Name: "test"}})
}
