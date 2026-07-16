// Package apikey defines the on-disk managed API-key registry shared by the
// gateway and an embedding product's local administration UI.
package apikey

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const registryVersion = 1

type Registry struct {
	Version int      `json:"version"`
	Keys    []Record `json:"keys"`
}

type Record struct {
	ID              string     `json:"id"`
	BudgetKey       string     `json:"budget_key"`
	Name            string     `json:"name"`
	Hash            string     `json:"hash"`
	Status          string     `json:"status"`
	DailyLimitUSD   float64    `json:"daily_limit_usd,omitempty"`
	MonthlyLimitUSD float64    `json:"monthly_limit_usd,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	RetireAt        *time.Time `json:"retire_at,omitempty"`
}

func Load(path string) (Registry, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Registry{Version: registryVersion}, nil
	}
	if err != nil {
		return Registry{}, fmt.Errorf("read managed API keys: %w", err)
	}
	var registry Registry
	if err := json.Unmarshal(data, &registry); err != nil {
		return Registry{}, fmt.Errorf("parse managed API keys: %w", err)
	}
	if registry.Version != registryVersion {
		return Registry{}, fmt.Errorf("managed API key registry version %d is unsupported", registry.Version)
	}
	return registry, nil
}

func Save(path string, registry Registry) error {
	registry.Version = registryVersion
	data, err := json.MarshalIndent(registry, "", "  ")
	if err != nil {
		return fmt.Errorf("encode managed API keys: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create managed API key directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".managed-keys-*")
	if err != nil {
		return fmt.Errorf("create managed API key file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return fmt.Errorf("protect managed API key file: %w", err)
	}
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		tmp.Close()
		return fmt.Errorf("write managed API keys: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("sync managed API keys: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close managed API keys: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace managed API keys: %w", err)
	}
	return nil
}

func GenerateToken() (string, error) {
	value := make([]byte, 32)
	if _, err := rand.Read(value); err != nil {
		return "", fmt.Errorf("generate API key: %w", err)
	}
	return "aion_" + base64.RawURLEncoding.EncodeToString(value), nil
}

func HashToken(token string) string {
	sum := sha256.Sum256([]byte("AION-MANAGED-API-KEY-V1\x00" + strings.TrimSpace(token)))
	return hex.EncodeToString(sum[:])
}

func Active(record Record, now time.Time) bool {
	if record.Status != "active" {
		return false
	}
	return record.RetireAt == nil || now.Before(*record.RetireAt)
}
