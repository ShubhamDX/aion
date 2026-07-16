package apikey

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRegistryStoresOnlyTokenHash(t *testing.T) {
	path := filepath.Join(t.TempDir(), "managed-keys.json")
	token, err := GenerateToken()
	if err != nil {
		t.Fatal(err)
	}
	record := Record{ID: "key-1", BudgetKey: "family-1", Name: "operator", Hash: HashToken(token), Status: "active", CreatedAt: time.Now().UTC()}
	if err := Save(path, Registry{Keys: []Record{record}}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), token) {
		t.Fatal("managed registry contains the raw API key")
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("registry mode = %o, want 600", info.Mode().Perm())
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Keys) != 1 || !Active(loaded.Keys[0], time.Now().UTC()) {
		t.Fatalf("loaded registry = %#v", loaded)
	}
}
