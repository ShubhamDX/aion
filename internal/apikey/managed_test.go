package apikey

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
	managed "github.com/ShubhamDX/aion/pkg/apikey"
)

func TestValidatorReloadsManagedKeysAndKeepsBudgetIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	token, err := managed.GenerateToken()
	if err != nil {
		t.Fatal(err)
	}
	record := managed.Record{ID: "key-1", BudgetKey: "family-1", Name: "intern", Hash: managed.HashToken(token), Status: "active", DailyLimitUSD: 3, CreatedAt: time.Now().UTC()}
	if err := managed.Save(path, managed.Registry{Keys: []managed.Record{record}}); err != nil {
		t.Fatal(err)
	}
	validator := NewValidator([]config.KeyConfig{}, path)
	info, ok := validator.Validate(token)
	if !ok || info.Key != "managed:family-1" || info.DailyLimitUSD != 3 {
		t.Fatalf("validation = %#v, %v", info, ok)
	}
	record.Status = "revoked"
	if err := managed.Save(path, managed.Registry{Keys: []managed.Record{record}}); err != nil {
		t.Fatal(err)
	}
	validator.mu.Lock()
	validator.nextReload = time.Time{}
	validator.mu.Unlock()
	if _, ok := validator.Validate(token); ok {
		t.Fatal("revoked managed key remained valid")
	}
}
