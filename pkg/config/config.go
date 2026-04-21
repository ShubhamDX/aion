package config

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
	OpenAI     *ProviderConfig      `yaml:"openai,omitempty"`
	Anthropic  *ProviderConfig      `yaml:"anthropic,omitempty"`
	OpenRouter *ProviderConfig      `yaml:"openrouter,omitempty"`
	Bedrock    *ProviderConfig      `yaml:"bedrock,omitempty"`
	Vertex     *ProviderConfig      `yaml:"vertex,omitempty"`
	Gemini     *ProviderConfig      `yaml:"gemini,omitempty"`
	Grok       *ProviderConfig      `yaml:"grok,omitempty"`
	Local      *LocalProviderConfig `yaml:"local,omitempty"`
}

// LocalProviderConfig configures the local llama.cpp inference provider.
type LocalProviderConfig struct {
	Enabled bool              `yaml:"enabled"`
	BaseURL string            `yaml:"base_url,omitempty"`
	Models  []ModelConfig     `yaml:"models"`
	Managed *ManagedLlamaConfig `yaml:"managed,omitempty"`
}

// ManagedLlamaConfig controls the managed llama-server subprocess (dev/single-node mode).
type ManagedLlamaConfig struct {
	BinaryPath   string   `yaml:"binary_path"`
	ModelPath    string   `yaml:"model_path"`
	Port         int      `yaml:"port,omitempty"`
	Threads      int      `yaml:"threads,omitempty"`
	ContextSize  int      `yaml:"ctx_size,omitempty"`
	GPULayers    int      `yaml:"gpu_layers,omitempty"`
	BatchSize    int      `yaml:"batch_size,omitempty"`
	ReadyTimeout string   `yaml:"ready_timeout,omitempty"`
	ExtraArgs    []string `yaml:"extra_args,omitempty"`
}

// ProviderConfig is the configuration for a single LLM provider.
type ProviderConfig struct {
	APIKey    string        `yaml:"api_key"`
	BaseURL   string        `yaml:"base_url,omitempty"`
	Region    string        `yaml:"region,omitempty"`
	ProjectID string        `yaml:"project_id,omitempty"`
	Models    []ModelConfig `yaml:"models"`
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

// TelemetryConfig controls the async telemetry recorder.
type TelemetryConfig struct {
	DBPath        string `yaml:"db_path"`
	BatchSize     int    `yaml:"batch_size"`
	FlushInterval string `yaml:"flush_interval"`
}
