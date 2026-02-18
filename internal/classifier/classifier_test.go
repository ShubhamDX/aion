package classifier

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
	"github.com/ShubhamDX/aion/models"
)

func TestEmbeddedDefaultModel(t *testing.T) {
	if len(models.DefaultIntentModelJSON) == 0 {
		t.Fatal("embedded default model is empty")
	}
	m, err := LoadIntentModelFromBytes(models.DefaultIntentModelJSON)
	if err != nil {
		t.Fatalf("failed to load embedded default model: %v", err)
	}
	if len(m.Categories) != 14 {
		t.Errorf("expected 14 categories, got %d", len(m.Categories))
	}

	// Smoke-test predictions.
	tests := []struct {
		input   string
		wantMin float64
		wantMax float64
	}{
		{"hi", 0.00, 0.35},
		{"what is the capital of France", 0.00, 0.30},
		{"design a distributed system at scale", 0.40, 1.00},
	}
	for _, tt := range tests {
		score := m.Predict(tt.input)
		if score < tt.wantMin || score > tt.wantMax {
			t.Errorf("Predict(%q) = %.4f, want [%.2f, %.2f]", tt.input, score, tt.wantMin, tt.wantMax)
		}
	}
}

// raw is a test helper that wraps a string as json.RawMessage.
func raw(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

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
			{Role: "user", Content: raw("what is 2+2")},
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
			{Role: "system", Content: raw("You are an expert assistant. Please analyze carefully and provide detailed step-by-step reasoning.")},
			{Role: "user", Content: raw("Hello, I need help with a project.")},
			{Role: "assistant", Content: raw("Sure, I'd be happy to help.")},
			{Role: "user", Content: raw("Can you explain how to design a REST API? Compare different approaches and evaluate their tradeoffs.")},
			{Role: "assistant", Content: raw("There are several approaches to consider.")},
			{Role: "user", Content: raw("Now analyze the performance implications and explain the tradeoffs. Compare REST vs GraphQL and evaluate which is better for this use case. Also help me debug this issue.")},
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
			{Role: "system", Content: raw(longSystemPrompt)},
			{Role: "user", Content: raw("Please analyze and refactor this code:\n```go\nfunc process(items []Item) {\n\tfor i := 0; i < len(items); i++ {\n\t\t// process\n\t}\n}\n```\nAlso implement a new optimized version, debug any issues, and evaluate the performance. Design an architecture that handles $10M requests per day.")},
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
			{Role: "user", Content: raw("hi")},
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
			{Role: "user", Content: raw("hello")},
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
			{Role: "user", Content: raw("hello")},
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

// --- Intent signal unit tests ---

func makeIntentReq(userMsg string) *types.ChatCompletionRequest {
	return &types.ChatCompletionRequest{
		Messages: []types.Message{
			{Role: "user", Content: raw(userMsg)},
		},
	}
}

func TestIntentSignal_Greeting(t *testing.T) {
	score := intentSignal(makeIntentReq("hi"))
	if score != 0.0 {
		t.Errorf("expected 0.0 for greeting, got %.4f", score)
	}
}

func TestIntentSignal_FactualLookup(t *testing.T) {
	score := intentSignal(makeIntentReq("what is the capital of France"))
	if score != 0.05 {
		t.Errorf("expected 0.05 for factual lookup, got %.4f", score)
	}
}

func TestIntentSignal_CodeGeneration(t *testing.T) {
	score := intentSignal(makeIntentReq("write a function to sort an array"))
	if score != 0.45 {
		t.Errorf("expected 0.45 for code generation, got %.4f", score)
	}
}

func TestIntentSignal_MultiStep(t *testing.T) {
	score := intentSignal(makeIntentReq("first, analyze the requirements then design the architecture"))
	if score < 0.75 {
		t.Errorf("expected >= 0.75 for multi-step task, got %.4f", score)
	}
}

func TestIntentSignal_ArchitectureDesign(t *testing.T) {
	score := intentSignal(makeIntentReq("design a system that handles millions of requests at scale"))
	if score < 0.80 {
		t.Errorf("expected >= 0.80 for architecture/design, got %.4f", score)
	}
}

// --- ML model-based intent tests ---

func TestIntentModel_LoadAndPredict(t *testing.T) {
	modelPath := "../../models/intent_classifier.json"
	model, err := LoadIntentModel(modelPath)
	if err != nil {
		t.Skipf("intent model not available at %s: %v", modelPath, err)
	}

	tests := []struct {
		input   string
		wantMin float64
		wantMax float64
	}{
		{"hi", 0.00, 0.35},
		{"what is 2+2", 0.00, 0.30},
		{"design a system that handles millions of users at scale", 0.50, 1.00},
		{"fix this bug in my login function", 0.30, 0.70},
		{"evaluate your own reasoning about this problem", 0.70, 1.00},
	}
	for _, tt := range tests {
		score := model.Predict(tt.input)
		if score < tt.wantMin || score > tt.wantMax {
			t.Errorf("Predict(%q) = %.4f, want [%.2f, %.2f]", tt.input, score, tt.wantMin, tt.wantMax)
		}
	}
}

func TestMakeIntentSignal_WithModel(t *testing.T) {
	modelPath := "../../models/intent_classifier.json"
	model, err := LoadIntentModel(modelPath)
	if err != nil {
		t.Skipf("intent model not available: %v", err)
	}

	fn := makeIntentSignal(model)

	// Architecture prompt should get a high score from the model + structural bonuses.
	req := makeIntentReq("design a system that handles millions of users at scale")
	score := fn(req)
	if score < 0.50 {
		t.Errorf("model-based intent score for architecture prompt = %.4f, expected >= 0.50", score)
	}

	// Greeting should stay low.
	reqHi := makeIntentReq("hi")
	scoreHi := fn(reqHi)
	if scoreHi > 0.30 {
		t.Errorf("model-based intent score for greeting = %.4f, expected <= 0.30", scoreHi)
	}
}
