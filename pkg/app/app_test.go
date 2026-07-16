package app

import (
	"path/filepath"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
)

func TestLoadConfigUsesProvidedConfig(t *testing.T) {
	want := &config.Config{}
	want.Providers.OpenAI = &config.ProviderConfig{Models: []config.ModelConfig{{ID: "test", Tier: 1}}}

	got, err := loadConfig(Options{
		Config:     want,
		ConfigPath: filepath.Join(t.TempDir(), "missing.yaml"),
	})
	if err != nil {
		t.Fatalf("load provided config: %v", err)
	}
	if got != want {
		t.Fatal("loadConfig returned a different config pointer")
	}
	if got.Server.Port != 8080 {
		t.Fatalf("server port = %d, want default 8080", got.Server.Port)
	}
}

func TestLoadConfigFallsBackToPath(t *testing.T) {
	_, err := loadConfig(Options{ConfigPath: filepath.Join(t.TempDir(), "missing.yaml")})
	if err == nil {
		t.Fatal("loadConfig must read ConfigPath when Config is nil")
	}
}
