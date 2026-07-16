// Package providercheck runs an operator-requested, metadata-only connection
// check against a configured model. It never returns model output or upstream
// error bodies.
package providercheck

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/provider"
	"github.com/ShubhamDX/aion/internal/types"
)

type Model struct {
	Provider       string `json:"provider"`
	ID             string `json:"id"`
	Tier           int    `json:"tier"`
	CredentialMode string `json:"credential_mode"`
}

type Result struct {
	Provider  string    `json:"provider"`
	Model     string    `json:"model"`
	Tier      int       `json:"tier"`
	Connected bool      `json:"connected"`
	Code      string    `json:"code"`
	Message   string    `json:"message"`
	LatencyMS int64     `json:"latency_ms"`
	CheckedAt time.Time `json:"checked_at"`
}

type Checker struct {
	providers map[string]provider.Provider
	models    []Model
}

func New(cfg *config.Config) (*Checker, error) {
	checker := &Checker{providers: make(map[string]provider.Provider)}
	add := func(name string, configured *config.ProviderConfig, instance provider.Provider) {
		if configured == nil || instance == nil {
			return
		}
		checker.providers[name] = instance
		mode := configured.CredentialMode
		if mode == "" {
			mode = "api_key"
			if name == "bedrock" && configured.APIKey == "" {
				mode = "aws_sdk"
			}
		}
		for _, model := range configured.Models {
			checker.models = append(checker.models, Model{Provider: name, ID: model.ID, Tier: model.Tier, CredentialMode: mode})
		}
	}
	add("openai", cfg.Providers.OpenAI, newProvider(cfg.Providers.OpenAI, provider.NewOpenAI))
	add("anthropic", cfg.Providers.Anthropic, newProvider(cfg.Providers.Anthropic, provider.NewAnthropic))
	add("openrouter", cfg.Providers.OpenRouter, newProvider(cfg.Providers.OpenRouter, provider.NewOpenRouter))
	if cfg.Providers.Bedrock != nil {
		instance, err := provider.NewBedrock(cfg.Providers.Bedrock)
		if err != nil {
			return nil, fmt.Errorf("configure Bedrock check: %w", err)
		}
		add("bedrock", cfg.Providers.Bedrock, instance)
	}
	add("vertex", cfg.Providers.Vertex, newProvider(cfg.Providers.Vertex, provider.NewVertex))
	add("gemini", cfg.Providers.Gemini, newProvider(cfg.Providers.Gemini, provider.NewGemini))
	add("grok", cfg.Providers.Grok, newProvider(cfg.Providers.Grok, provider.NewGrok))
	if local := cfg.Providers.Local; local != nil && local.Enabled {
		checker.providers["local"] = provider.NewLocal(local)
		for _, model := range local.Models {
			checker.models = append(checker.models, Model{Provider: "local", ID: model.ID, Tier: model.Tier, CredentialMode: "local"})
		}
	}
	sort.Slice(checker.models, func(i, j int) bool {
		if checker.models[i].Tier != checker.models[j].Tier {
			return checker.models[i].Tier < checker.models[j].Tier
		}
		if checker.models[i].Provider != checker.models[j].Provider {
			return checker.models[i].Provider < checker.models[j].Provider
		}
		return checker.models[i].ID < checker.models[j].ID
	})
	return checker, nil
}

func newProvider[T provider.Provider](cfg *config.ProviderConfig, build func(*config.ProviderConfig) T) provider.Provider {
	if cfg == nil {
		return nil
	}
	return build(cfg)
}

func (c *Checker) Models() []Model {
	return append([]Model(nil), c.models...)
}

func (c *Checker) Test(ctx context.Context, providerName, modelID string) Result {
	started := time.Now()
	result := Result{Provider: providerName, Model: modelID, CheckedAt: started.UTC()}
	for _, model := range c.models {
		if model.Provider == providerName && model.ID == modelID {
			result.Tier = model.Tier
			break
		}
	}
	instance, ok := c.providers[providerName]
	if !ok || result.Tier == 0 {
		result.Code = "not_configured"
		result.Message = "The selected provider and model are not configured."
		return result
	}
	maxTokens := 1
	content, _ := json.Marshal("Reply with OK.")
	request := &types.ChatCompletionRequest{
		Model: modelID, MaxTokens: &maxTokens,
		Messages: []types.Message{{Role: "user", Content: content}},
	}
	_, err := instance.Send(ctx, request, modelID)
	result.LatencyMS = time.Since(started).Milliseconds()
	if err == nil {
		result.Connected = true
		result.Code = "connected"
		result.Message = "The model accepted a one-token diagnostic request."
		return result
	}
	result.Code, result.Message = classifyError(err)
	return result
}

func classifyError(err error) (string, string) {
	message := strings.ToLower(err.Error())
	switch {
	case strings.Contains(message, "401"), strings.Contains(message, "403"), strings.Contains(message, "unauthorized"), strings.Contains(message, "accessdenied"):
		return "authentication_failed", "The provider rejected the configured credentials or model permission."
	case strings.Contains(message, "404"), strings.Contains(message, "not found"), strings.Contains(message, "validationexception"):
		return "model_unavailable", "The configured model is unavailable in this provider account or region."
	case strings.Contains(message, "429"), strings.Contains(message, "throttl"), strings.Contains(message, "quota"):
		return "quota_or_rate_limit", "The provider quota or rate limit blocked the diagnostic request."
	case strings.Contains(message, "deadline"), strings.Contains(message, "timeout"):
		return "timeout", "The provider did not respond before the diagnostic timeout."
	default:
		return "connection_failed", "The provider connection failed. Check the local gateway logs for the operator-only error."
	}
}
