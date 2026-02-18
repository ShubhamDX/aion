package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the top-level AION gateway configuration.
type Config struct {
	Server    ServerConfig    `yaml:"server"`
	Auth      AuthConfig      `yaml:"auth"`
	Providers ProvidersConfig `yaml:"providers"`
	Routing   RoutingConfig   `yaml:"routing"`
	Telemetry TelemetryConfig `yaml:"telemetry"`
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	Port            int    `yaml:"port"`
	ReadTimeout     string `yaml:"read_timeout"`
	WriteTimeout    string `yaml:"write_timeout"`
	ShutdownTimeout string `yaml:"shutdown_timeout"`
}

// AuthConfig controls API-key authentication.
type AuthConfig struct {
	Enabled bool        `yaml:"enabled"`
	Keys    []KeyConfig `yaml:"keys"`
}

// KeyConfig represents a single API key with optional budget limits.
type KeyConfig struct {
	Key    string       `yaml:"key"`
	Name   string       `yaml:"name"`
	Budget BudgetConfig `yaml:"budget"`
}

// BudgetConfig defines spending limits for an API key.
type BudgetConfig struct {
	DailyLimitUSD   float64 `yaml:"daily_limit_usd"`
	MonthlyLimitUSD float64 `yaml:"monthly_limit_usd"`
}

// ProvidersConfig holds configuration for each upstream LLM provider.
type ProvidersConfig struct {
	OpenAI     *ProviderConfig `yaml:"openai,omitempty"`
	Anthropic  *ProviderConfig `yaml:"anthropic,omitempty"`
	OpenRouter *ProviderConfig `yaml:"openrouter,omitempty"`
}

// ProviderConfig is the configuration for a single LLM provider.
type ProviderConfig struct {
	APIKey  string        `yaml:"api_key"`
	BaseURL string        `yaml:"base_url,omitempty"`
	Models  []ModelConfig `yaml:"models"`
}

// ModelConfig describes a model exposed through a provider.
type ModelConfig struct {
	ID               string  `yaml:"id"`
	Tier             int     `yaml:"tier"`
	InputPricePer1M  float64 `yaml:"input_price_per_1m"`
	OutputPricePer1M float64 `yaml:"output_price_per_1m"`
	MaxTokens        int     `yaml:"max_tokens,omitempty"`
}

// RoutingConfig controls how requests are routed to models.
type RoutingConfig struct {
	Strategy        string           `yaml:"strategy"`
	Classifier      ClassifierConfig `yaml:"classifier"`
	FallbackEnabled bool             `yaml:"fallback_enabled"`
}

// ClassifierConfig holds thresholds for the complexity classifier.
type ClassifierConfig struct {
	Tier1Threshold  float64 `yaml:"tier1_threshold"`
	Tier2Threshold  float64 `yaml:"tier2_threshold"`
	IntentModelPath string  `yaml:"intent_model_path,omitempty"`
}

// TelemetryConfig controls the async telemetry recorder.
type TelemetryConfig struct {
	DBPath        string `yaml:"db_path"`
	BatchSize     int    `yaml:"batch_size"`
	FlushInterval string `yaml:"flush_interval"`
}

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
