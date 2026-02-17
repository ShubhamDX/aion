package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// Server wraps an http.Server with graceful shutdown support.
type Server struct {
	httpServer *http.Server
	handler    http.Handler
}

// New creates a Server bound to the given port with the provided handler and timeouts.
func New(port int, handler http.Handler, readTimeout, writeTimeout time.Duration) *Server {
	return &Server{
		httpServer: &http.Server{
			Addr:         fmt.Sprintf(":%d", port),
			Handler:      handler,
			ReadTimeout:  readTimeout,
			WriteTimeout: writeTimeout,
		},
		handler: handler,
	}
}

// Start begins listening and serving HTTP requests. It blocks until the server
// is shut down or an error occurs.
func (s *Server) Start() error {
	slog.Info("starting AION server", "addr", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server without interrupting active connections.
func (s *Server) Shutdown(ctx context.Context) error {
	slog.Info("shutting down server")
	return s.httpServer.Shutdown(ctx)
}
