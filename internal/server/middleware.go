package server

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/types"
)

type contextKey string

const (
	// RequestIDKey is the context key for the unique request identifier.
	RequestIDKey contextKey = "request_id"
	// APIKeyInfoKey is the context key for the validated API key information.
	APIKeyInfoKey contextKey = "api_key_info"
)

// RequestID returns the request ID stored in the context, or an empty string
// if none is present.
func RequestID(ctx context.Context) string {
	if id, ok := ctx.Value(RequestIDKey).(string); ok {
		return id
	}
	return ""
}

// APIKeyInfo returns the API key info stored in the context, or nil if none
// is present.
func APIKeyInfo(ctx context.Context) *apikey.KeyInfo {
	if info, ok := ctx.Value(APIKeyInfoKey).(*apikey.KeyInfo); ok {
		return info
	}
	return nil
}

// statusRecorder wraps http.ResponseWriter to capture the status code written
// by downstream handlers.
type statusRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.statusCode = code
	r.ResponseWriter.WriteHeader(code)
}

// Flush implements http.Flusher so SSE streaming works through the middleware.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// RequestIDMiddleware adds a unique UUID request ID to each request context and
// sets the X-Request-ID response header.
func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := uuid.New().String()
		ctx := context.WithValue(r.Context(), RequestIDKey, id)
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// LoggingMiddleware logs the method, path, status code, and duration of each
// request at Info level.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(rec, r)
		slog.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.statusCode,
			"duration", time.Since(start),
			"request_id", RequestID(r.Context()),
		)
	})
}

// CORSMiddleware adds permissive CORS headers for development and handles
// OPTIONS preflight requests.
func CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// PanicRecoveryMiddleware catches panics in downstream handlers and returns a
// 500 Internal Server Error instead of crashing the process.
func PanicRecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				slog.Error("panic recovered",
					"error", err,
					"method", r.Method,
					"path", r.URL.Path,
					"request_id", RequestID(r.Context()),
				)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				json.NewEncoder(w).Encode(types.ErrorResponse{
					Error: types.ErrorDetail{
						Message: "internal server error",
						Type:    "server_error",
						Code:    "internal_error",
					},
				})
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// AuthMiddleware validates API keys from the Authorization header using the
// given validator. If validator is nil, authentication is skipped and all
// requests are allowed through.
func AuthMiddleware(validator *apikey.Validator) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if validator == nil {
				next.ServeHTTP(w, r)
				return
			}

			token := apikey.ExtractBearerToken(r.Header.Get("Authorization"))
			if token == "" {
				writeAuthError(w, "missing or invalid Authorization header")
				return
			}

			info, ok := validator.Validate(token)
			if !ok {
				writeAuthError(w, "invalid API key")
				return
			}

			ctx := context.WithValue(r.Context(), APIKeyInfoKey, info)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func writeAuthError(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	json.NewEncoder(w).Encode(types.ErrorResponse{
		Error: types.ErrorDetail{
			Message: message,
			Type:    "authentication_error",
			Code:    "invalid_api_key",
		},
	})
}

// Chain applies middleware in order where the first middleware in the list is
// the outermost (executed first on the request, last on the response).
func Chain(h http.Handler, middlewares ...func(http.Handler) http.Handler) http.Handler {
	for i := len(middlewares) - 1; i >= 0; i-- {
		h = middlewares[i](h)
	}
	return h
}
