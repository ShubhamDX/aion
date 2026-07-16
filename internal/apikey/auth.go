package apikey

import (
	"strings"
	"sync"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	managed "github.com/ShubhamDX/aion/pkg/apikey"
)

// KeyInfo holds validated information about an API key.
type KeyInfo struct {
	Key             string
	Name            string
	DailyLimitUSD   float64
	MonthlyLimitUSD float64
}

// Validator performs API key lookups against a pre-built map.
type Validator struct {
	keys        map[string]*KeyInfo
	managedPath string
	mu          sync.RWMutex
	managed     map[string]*KeyInfo
	nextReload  time.Time
}

// NewValidator builds a Validator from the supplied key configurations.
func NewValidator(keys []config.KeyConfig, managedPath ...string) *Validator {
	m := make(map[string]*KeyInfo, len(keys))
	for _, k := range keys {
		m[k.Key] = &KeyInfo{
			Key:             k.Key,
			Name:            k.Name,
			DailyLimitUSD:   k.Budget.DailyLimitUSD,
			MonthlyLimitUSD: k.Budget.MonthlyLimitUSD,
		}
	}
	path := ""
	if len(managedPath) > 0 {
		path = strings.TrimSpace(managedPath[0])
	}
	return &Validator{keys: m, managedPath: path, managed: make(map[string]*KeyInfo)}
}

// Validate looks up an API key and returns the associated KeyInfo.
// The second return value indicates whether the key was found.
func (v *Validator) Validate(key string) (*KeyInfo, bool) {
	if info, ok := v.keys[key]; ok {
		return info, true
	}
	if v.managedPath == "" {
		return nil, false
	}
	v.reloadManaged(time.Now().UTC())
	hash := managed.HashToken(key)
	v.mu.RLock()
	info, ok := v.managed[hash]
	v.mu.RUnlock()
	return info, ok
}

func (v *Validator) reloadManaged(now time.Time) {
	v.mu.RLock()
	if now.Before(v.nextReload) {
		v.mu.RUnlock()
		return
	}
	v.mu.RUnlock()

	registry, err := managed.Load(v.managedPath)
	next := now.Add(time.Second)
	loaded := make(map[string]*KeyInfo)
	if err == nil {
		for _, record := range registry.Keys {
			if !managed.Active(record, now) {
				continue
			}
			budgetKey := record.BudgetKey
			if budgetKey == "" {
				budgetKey = record.ID
			}
			loaded[record.Hash] = &KeyInfo{
				Key: "managed:" + budgetKey, Name: record.Name,
				DailyLimitUSD: record.DailyLimitUSD, MonthlyLimitUSD: record.MonthlyLimitUSD,
			}
		}
	}
	v.mu.Lock()
	if err == nil {
		v.managed = loaded
	}
	v.nextReload = next
	v.mu.Unlock()
}

// ExtractBearerToken extracts the token from an Authorization header
// value of the form "Bearer <token>". It returns an empty string if
// the header does not use the Bearer scheme.
func ExtractBearerToken(authHeader string) string {
	const prefix = "Bearer "
	if !strings.HasPrefix(authHeader, prefix) {
		return ""
	}
	return strings.TrimSpace(authHeader[len(prefix):])
}
