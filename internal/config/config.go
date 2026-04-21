package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	pkgconfig "github.com/ShubhamDX/aion/pkg/config"
)

// Type aliases — all existing internal imports compile unchanged.
type Config = pkgconfig.Config
type ServerConfig = pkgconfig.ServerConfig
type AuthConfig = pkgconfig.AuthConfig
type KeyConfig = pkgconfig.KeyConfig
type BudgetConfig = pkgconfig.BudgetConfig
type ProvidersConfig = pkgconfig.ProvidersConfig
type ProviderConfig = pkgconfig.ProviderConfig
type ModelConfig = pkgconfig.ModelConfig
type RoutingConfig = pkgconfig.RoutingConfig
type ClassifierConfig = pkgconfig.ClassifierConfig
type TelemetryConfig = pkgconfig.TelemetryConfig
type LocalProviderConfig = pkgconfig.LocalProviderConfig
type ManagedLlamaConfig = pkgconfig.ManagedLlamaConfig

// Load reads the YAML configuration file at the given path, expands
// environment variables in the raw content, unmarshals the result,
// applies sensible defaults, and validates the final configuration.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("config: read file: %w", err)
	}

	// Expand environment variables ($VAR and ${VAR} patterns).
	expanded := os.ExpandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("config: parse yaml: %w", err)
	}

	applyDefaults(&cfg)

	if err := validate(&cfg); err != nil {
		return nil, fmt.Errorf("config: validation: %w", err)
	}

	return &cfg, nil
}
