package classifier

import (
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

func newTestClassifier() *Classifier {
	return New(config.ClassifierConfig{
		Tier1Threshold: 0.35,
		Tier2Threshold: 0.70,
	})
}

func TestSimpleRequest_Tier1(t *testing.T) {
	c := newTestClassifier()
	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "user", Content: "what is 2+2"},
		},
	}
	tier, score, signals := c.Classify(req)
	if tier != types.Tier1 {
		t.Errorf("expected Tier1, got %v (score=%.4f, signals=%v)", tier, score, signals)
	}
	if score >= 0.35 {
		t.Errorf("expected score < 0.35, got %.4f", score)
	}
}

func TestMediumRequest_Tier2(t *testing.T) {
	c := newTestClassifier()
	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "system", Content: "You are an expert assistant. Please analyze carefully and provide detailed step-by-step reasoning."},
			{Role: "user", Content: "Hello, I need help with a project."},
			{Role: "assistant", Content: "Sure, I'd be happy to help."},
			{Role: "user", Content: "Can you explain how to design a REST API? Compare different approaches and evaluate their tradeoffs."},
			{Role: "assistant", Content: "There are several approaches to consider."},
			{Role: "user", Content: "Now analyze the performance implications and explain the tradeoffs. Compare REST vs GraphQL and evaluate which is better for this use case. Also help me debug this issue."},
		},
		Tools: []types.Tool{
			{Type: "function", Function: types.FunctionDef{Name: "search", Description: "search the web"}},
			{Type: "function", Function: types.FunctionDef{Name: "fetch", Description: "fetch a URL"}},
		},
	}
	tier, score, signals := c.Classify(req)
	if tier != types.Tier2 {
		t.Errorf("expected Tier2, got %v (score=%.4f, signals=%v)", tier, score, signals)
	}
}

func TestComplexRequest_Tier3(t *testing.T) {
	c := newTestClassifier()

	longSystemPrompt := "You are an expert software architect. " +
		"Analyze the following code step-by-step using chain of thought reasoning. " +
		"Provide detailed explanations for each decision. " +
		strings.Repeat("Consider edge cases and performance implications. ", 40)

	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "system", Content: longSystemPrompt},
			{Role: "user", Content: "Please analyze and refactor this code:\n```go\nfunc process(items []Item) {\n\tfor i := 0; i < len(items); i++ {\n\t\t// process\n\t}\n}\n```\nAlso implement a new optimized version, debug any issues, and evaluate the performance. Design an architecture that handles $10M requests per day."},
		},
		Tools: []types.Tool{
			{Type: "function", Function: types.FunctionDef{Name: "search"}},
			{Type: "function", Function: types.FunctionDef{Name: "read_file"}},
			{Type: "function", Function: types.FunctionDef{Name: "write_file"}},
			{Type: "function", Function: types.FunctionDef{Name: "run_tests"}},
			{Type: "function", Function: types.FunctionDef{Name: "deploy"}},
		},
		AIONPreferences: &types.AIONPreferences{
			PreferQuality: true,
		},
	}
	tier, score, signals := c.Classify(req)
	if tier != types.Tier3 {
		t.Errorf("expected Tier3, got %v (score=%.4f, signals=%v)", tier, score, signals)
	}
}

func TestUserHintOverride(t *testing.T) {
	c := newTestClassifier()
	preferredTier := 3
	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "user", Content: "hi"},
		},
		AIONPreferences: &types.AIONPreferences{
			PreferredTier: &preferredTier,
		},
	}
	tier, score, signals := c.Classify(req)
	// With preferred_tier=3 the user_hints signal returns 1.0 (weight 0.15 -> +0.15).
	// A simple "hi" message should still be low on other signals.
	// The boost should push the score higher than a plain "hi" without hints.
	if signals["user_hints"] != 1.0 {
		t.Errorf("expected user_hints signal = 1.0, got %.4f", signals["user_hints"])
	}
	t.Logf("UserHintOverride: tier=%v score=%.4f signals=%v", tier, score, signals)
}

func TestEmptyRequest_Tier1(t *testing.T) {
	c := newTestClassifier()
	req := &types.ChatCompletionRequest{}
	tier, score, signals := c.Classify(req)
	if tier != types.Tier1 {
		t.Errorf("expected Tier1 for empty request, got %v (score=%.4f, signals=%v)", tier, score, signals)
	}
}

func TestTierFromScore(t *testing.T) {
	tests := []struct {
		score float64
		want  types.Tier
	}{
		{0.0, types.Tier1},
		{0.20, types.Tier1},
		{0.34, types.Tier1},
		{0.35, types.Tier2},
		{0.50, types.Tier2},
		{0.69, types.Tier2},
		{0.70, types.Tier3},
		{0.85, types.Tier3},
		{1.0, types.Tier3},
	}
	for _, tt := range tests {
		got := TierFromScore(tt.score, 0.35, 0.70)
		if got != tt.want {
			t.Errorf("TierFromScore(%.2f) = %v, want %v", tt.score, got, tt.want)
		}
	}
}

func TestPreferQualityHint(t *testing.T) {
	c := newTestClassifier()
	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "user", Content: "hello"},
		},
		AIONPreferences: &types.AIONPreferences{
			PreferQuality: true,
		},
	}
	_, _, signals := c.Classify(req)
	if signals["user_hints"] != 0.8 {
		t.Errorf("expected user_hints = 0.8 for PreferQuality, got %.4f", signals["user_hints"])
	}
}

func TestPreferSpeedHint(t *testing.T) {
	c := newTestClassifier()
	req := &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "user", Content: "hello"},
		},
		AIONPreferences: &types.AIONPreferences{
			PreferSpeed: true,
		},
	}
	_, _, signals := c.Classify(req)
	if signals["user_hints"] != 0.1 {
		t.Errorf("expected user_hints = 0.1 for PreferSpeed, got %.4f", signals["user_hints"])
	}
}
