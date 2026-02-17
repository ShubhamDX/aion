package apikey

import (
	"strings"

	"github.com/ShubhamDX/aion/internal/config"
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
	keys map[string]*KeyInfo
}

// NewValidator builds a Validator from the supplied key configurations.
func NewValidator(keys []config.KeyConfig) *Validator {
	m := make(map[string]*KeyInfo, len(keys))
	for _, k := range keys {
		m[k.Key] = &KeyInfo{
			Key:             k.Key,
			Name:            k.Name,
			DailyLimitUSD:   k.Budget.DailyLimitUSD,
			MonthlyLimitUSD: k.Budget.MonthlyLimitUSD,
		}
	}
	return &Validator{keys: m}
}

// Validate looks up an API key and returns the associated KeyInfo.
// The second return value indicates whether the key was found.
func (v *Validator) Validate(key string) (*KeyInfo, bool) {
	info, ok := v.keys[key]
	return info, ok
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
