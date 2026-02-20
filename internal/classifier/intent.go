package classifier

import (
	"math"
	"regexp"
	"strings"

	"github.com/ShubhamDX/aion/internal/types"
)

// systemReminderRe matches <system-reminder>...</system-reminder> blocks
// injected by clients like Claude Code into user messages.
var systemReminderRe = regexp.MustCompile(`(?s)<system-reminder>.*?</system-reminder>`)

// stripSystemReminders removes all <system-reminder> blocks from content
// and trims the result.
func stripSystemReminders(s string) string {
	return strings.TrimSpace(systemReminderRe.ReplaceAllString(s, ""))
}

// makeIntentSignal returns an intentSignal function. If a trained model is
// provided the base intent score comes from the model; otherwise it falls back
// to pattern-matching heuristics.
func makeIntentSignal(model *IntentModel) SignalFunc {
	return func(req *types.ChatCompletionRequest) float64 {
		return intentSignalWithModel(req, model)
	}
}

// intentSignal is the default (pattern-only) entry point, kept for tests.
func intentSignal(req *types.ChatCompletionRequest) float64 {
	return intentSignalWithModel(req, nil)
}

// intentSignalWithModel analyses user messages to determine the type and
// structural complexity of the task being requested, returning 0.0 to 1.0.
func intentSignalWithModel(req *types.ChatCompletionRequest, model *IntentModel) float64 {
	if req == nil {
		return 0.0
	}

	// Use only the last user message — earlier user messages may contain
	// tool results or prior context that doesn't reflect current complexity.
	var content string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if strings.ToLower(req.Messages[i].Role) == "user" {
			content = strings.TrimSpace(req.Messages[i].ContentString())
			break
		}
	}

	// Strip <system-reminder>...</system-reminder> tags injected by clients
	// like Claude Code — these are scaffolding, not user intent.
	content = stripSystemReminders(content)
	if content == "" {
		return 0.0
	}

	lower := strings.ToLower(content)

	// Short messages (< 20 chars) are almost always simple — greetings,
	// quick questions, etc. Cap the score to avoid misclassification by
	// the ML model on out-of-distribution short inputs.
	shortMessage := len(content) < 20

	var baseScore float64
	if model != nil {
		baseScore = model.Predict(content)
	} else {
		baseScore = detectPrimaryIntent(lower)
	}

	if shortMessage && baseScore > 0.2 {
		baseScore = 0.2
	}
	taskBonus := countDistinctTasks(lower)
	conditionalBonus, enumerationBonus := detectStructuralComplexity(lower)
	depthMultiplier := questionDepthMultiplier(lower)

	score := baseScore + taskBonus + conditionalBonus + enumerationBonus
	score = math.Min(1.0, score)
	score *= depthMultiplier
	return math.Min(1.0, score)
}

// detectPrimaryIntent scans the content for known intent patterns and returns
// the base complexity score of the highest-scoring match.
func detectPrimaryIntent(lower string) float64 {
	var best float64

	// Meta-cognitive (1.00)
	if containsAny(lower, "evaluate your own", "reason about reasoning", "reflect on your") {
		best = math.Max(best, 1.00)
	}

	// Deep Reasoning (0.90)
	if containsAny(lower, "prove that", "derive the", "formal proof", "mathematical proof") {
		best = math.Max(best, 0.90)
	}

	// Architecture/Design (0.80)
	if containsAny(lower, "design a system", "architect", "at scale", "system design") {
		best = math.Max(best, 0.80)
	}

	// Multi-step Task (0.75)
	if hasMultiStepIndicators(lower) {
		best = math.Max(best, 0.75)
	}

	// Analysis (0.60)
	if containsAny(lower, "analyze", "evaluate", "assess", "review") {
		best = math.Max(best, 0.60)
	}

	// Debugging (0.55)
	if containsAny(lower, "debug", "fix ", "why is this not working", "not working", "error") {
		best = math.Max(best, 0.55)
	}

	// Comparison (0.50)
	if containsAny(lower, "compare", "difference between", "pros and cons", " vs ") {
		best = math.Max(best, 0.50)
	}

	// Code Generation (0.45)
	if containsAny(lower, "write a function", "create a class", "implement", "write code") {
		best = math.Max(best, 0.45)
	}

	// Explanation (0.30)
	if containsAny(lower, "explain", "how does", "why does", "how do") {
		best = math.Max(best, 0.30)
	}

	// Summarization (0.30)
	if containsAny(lower, "summarize", "tl;dr", "tldr", "give me the gist") {
		best = math.Max(best, 0.30)
	}

	// Simple Generation (0.15) — only when no higher generation intent matched.
	if best < 0.45 && containsAny(lower, "write a ", "suggest ", "come up with") {
		best = math.Max(best, 0.15)
	}

	// Translation/Format (0.10)
	if containsAny(lower, "translate", "convert ", "format as", "rewrite in") {
		best = math.Max(best, 0.10)
	}

	// Factual Lookup (0.05)
	if containsAny(lower, "what is", "who is", "when did", "define ", "list the", "what are") {
		best = math.Max(best, 0.05)
	}

	// Greeting/Chitchat (0.00) — the default zero value handles this.

	return best
}

// hasMultiStepIndicators detects sequential or enumerated task structures.
func hasMultiStepIndicators(lower string) bool {
	if strings.Contains(lower, "1.") && strings.Contains(lower, "2.") {
		return true
	}
	if strings.Contains(lower, "step 1") || strings.Contains(lower, "step 2") {
		return true
	}
	if strings.Contains(lower, "first") && strings.Contains(lower, "then") {
		return true
	}
	return false
}

// taskVerbs are imperative verbs that typically indicate distinct tasks.
var taskVerbs = []string{
	"analyze", "explain", "compare", "prove", "derive",
	"implement", "architect", "design", "optimize", "refactor",
	"debug", "evaluate", "fix", "write", "create", "build",
	"review", "assess", "summarize", "translate", "convert",
	"deploy", "test",
}

// countDistinctTasks counts distinct task verbs in the content and returns a
// bonus of +0.15 per verb beyond the first, capped at +0.30.
func countDistinctTasks(lower string) float64 {
	count := 0
	for _, verb := range taskVerbs {
		if strings.Contains(lower, verb) {
			count++
		}
	}

	if count <= 1 {
		return 0.0
	}
	bonus := float64(count-1) * 0.15
	return math.Min(0.30, bonus)
}

// detectStructuralComplexity returns bonuses for conditional constraints and
// enumerated structures found in the content.
func detectStructuralComplexity(lower string) (conditionalBonus, enumerationBonus float64) {
	if containsAny(lower, "if ", "assuming ", "given that", "under the constraint", "considering ") {
		conditionalBonus = 0.10
	}

	if containsAny(lower, "1.", "2.", "first,", "second,", "step 1") || strings.Contains(lower, "\n- ") {
		enumerationBonus = 0.10
	}

	return conditionalBonus, enumerationBonus
}

// questionDepthMultiplier returns a scaling factor based on the depth of
// interrogatives in the content.
//
//	Simple (what/who/when/where): ×1.0
//	Deep   (how/why):             ×1.2
//	Compound (2+ of how/why/what-if): ×1.4
func questionDepthMultiplier(lower string) float64 {
	deepCount := 0
	if strings.Contains(lower, "how ") {
		deepCount++
	}
	if strings.Contains(lower, "why ") {
		deepCount++
	}
	if containsAny(lower, "what if", "what-if") {
		deepCount++
	}

	switch {
	case deepCount >= 2:
		return 1.4
	case deepCount == 1:
		return 1.2
	default:
		return 1.0
	}
}

// containsAny reports whether s contains any of the given substrings.
func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
