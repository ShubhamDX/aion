package provider

import (
	"sync"
	"time"
)

// CircuitState represents the state of a circuit breaker.
type CircuitState int

const (
	StateClosed   CircuitState = iota // healthy, requests pass through
	StateOpen                         // unhealthy, requests blocked
	StateHalfOpen                     // testing, one request allowed
)

const (
	defaultFailureThreshold = 3
	defaultCooldown         = 60 * time.Second
)

type circuitBreaker struct {
	state       CircuitState
	failures    int
	threshold   int
	lastFailure time.Time
	cooldown    time.Duration
	mu          sync.RWMutex
}

func newCircuitBreaker() *circuitBreaker {
	return &circuitBreaker{
		state:     StateClosed,
		threshold: defaultFailureThreshold,
		cooldown:  defaultCooldown,
	}
}

// HealthMonitor tracks circuit breaker state per provider:model pair.
type HealthMonitor struct {
	breakers map[string]*circuitBreaker
	mu       sync.RWMutex
}

// NewHealthMonitor creates a new HealthMonitor.
func NewHealthMonitor() *HealthMonitor {
	return &HealthMonitor{
		breakers: make(map[string]*circuitBreaker),
	}
}

func breakerKey(provider, model string) string {
	return provider + ":" + model
}

func (h *HealthMonitor) getOrCreate(provider, model string) *circuitBreaker {
	key := breakerKey(provider, model)

	h.mu.RLock()
	cb, ok := h.breakers[key]
	h.mu.RUnlock()
	if ok {
		return cb
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	// Double-check after acquiring write lock.
	if cb, ok = h.breakers[key]; ok {
		return cb
	}
	cb = newCircuitBreaker()
	h.breakers[key] = cb
	return cb
}

// IsHealthy returns true if requests should be allowed for the given provider and model.
func (h *HealthMonitor) IsHealthy(provider, model string) bool {
	cb := h.getOrCreate(provider, model)

	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case StateClosed:
		return true
	case StateOpen:
		if time.Since(cb.lastFailure) >= cb.cooldown {
			cb.state = StateHalfOpen
			return true
		}
		return false
	case StateHalfOpen:
		return true
	default:
		return true
	}
}

// RecordSuccess resets the circuit breaker for the given provider and model.
func (h *HealthMonitor) RecordSuccess(provider, model string) {
	cb := h.getOrCreate(provider, model)

	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failures = 0
	cb.state = StateClosed
}

// RecordFailure increments the failure count and trips the breaker if the threshold is reached.
func (h *HealthMonitor) RecordFailure(provider, model string) {
	cb := h.getOrCreate(provider, model)

	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failures++
	cb.lastFailure = time.Now()
	if cb.failures >= cb.threshold {
		cb.state = StateOpen
	}
}
