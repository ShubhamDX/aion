package classifier

import (
	"math"
	"strings"

	"github.com/ShubhamDX/aion/internal/types"
)

// SignalFunc extracts a complexity signal from a request, returning 0.0-1.0.
type SignalFunc func(req *types.ChatCompletionRequest) float64

// tokenVolumeSignal estimates token count from non-system user/assistant
// messages (excluding the constant system prompt scaffolding) and scores
// against a 4000-token reference.
// Weight: 0.15
func tokenVolumeSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}
	totalChars := 0
	for _, msg := range req.Messages {
		if strings.ToLower(msg.Role) == "system" {
			continue
		}
		totalChars += len(msg.ContentString())
	}
	estimatedTokens := float64(totalChars) / 4.0
	return math.Min(1.0, estimatedTokens/4000.0)
}

// messageCountSignal scores the number of messages against a 20-message reference.
// Weight: 0.10
func messageCountSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}
	return math.Min(1.0, float64(len(req.Messages))/20.0)
}

// systemPromptSignal evaluates the system message for complexity keywords only
// (not length). Many clients like Claude Code send large constant system
// prompts with every request — scoring by length would always push to Tier 3.
// Weight: 0.10
func systemPromptSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}

	var systemContent string
	for _, msg := range req.Messages {
		if strings.ToLower(msg.Role) == "system" {
			systemContent = msg.ContentString()
			break
		}
	}

	if systemContent == "" {
		return 0.0
	}

	lower := strings.ToLower(systemContent)

	// Only score on strong complexity keywords, not length.
	keywords := []string{"step-by-step", "chain of thought", "formal proof", "mathematical"}
	matchCount := 0
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			matchCount++
		}
	}

	return math.Min(1.0, float64(matchCount)*0.3)
}

// toolPresenceSignal gives a small bump when tools are present but doesn't
// scale linearly — many clients (Claude Code, Cursor) always send 10-30
// tools as scaffolding. A binary "tools present" signal is more useful.
// Weight: 0.10
func toolPresenceSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil || len(req.Tools) == 0 {
		return 0.0
	}
	return 0.3
}

// contentKeywordsSignal checks the last user message for complexity indicators.
// Only the last user message is used because earlier messages may contain
// tool results, structured data, or other non-indicative content.
// Weight: 0.25
func contentKeywordsSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}

	// Use only the last user message for keyword analysis, stripping
	// <system-reminder> tags injected by clients like Claude Code.
	var content string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if strings.ToLower(req.Messages[i].Role) == "user" {
			content = stripSystemReminders(req.Messages[i].ContentString())
			break
		}
	}

	if content == "" {
		return 0.0
	}

	lower := strings.ToLower(content)

	keywords := []string{
		"analyze", "explain", "compare", "prove", "derive",
		"implement", "architect", "design", "optimize", "refactor",
		"debug", "evaluate",
	}
	matchCount := 0
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			matchCount++
		}
	}

	var codeBlockBonus float64
	if strings.Contains(lower, "```") {
		codeBlockBonus = 0.2
	}

	var mathBonus float64
	if strings.Contains(lower, "$") || strings.Contains(lower, "\\equation") {
		mathBonus = 0.15
	}

	// Also check for JSON structures.
	if strings.Contains(lower, "{") && strings.Contains(lower, "}") {
		codeBlockBonus = math.Max(codeBlockBonus, 0.1)
	}

	score := float64(matchCount)*0.20 + codeBlockBonus + mathBonus
	return math.Min(1.0, score)
}

// confirmationPatterns are short messages that indicate the user is confirming
// a previous assistant proposal rather than making a new request.
var confirmationPatterns = []string{
	"do it", "yes", "go ahead", "proceed", "sure", "ok", "okay", "yep", "yeah",
	"please do", "go for it", "sounds good", "that works", "let's do it",
	"make it so", "approved", "lgtm", "ship it", "yes please", "yes go ahead",
	"do that", "go", "continue", "yes do it", "implement it", "do this",
	"let's go", "make it happen", "start", "begin", "run it", "y", "yea",
	"implement this", "make the change", "make the changes", "apply it",
	"go ahead and do it", "please proceed", "do it please",
}

// isConfirmation returns true if the stripped, lowered user message is a
// short affirmative that confirms a prior assistant proposal.
func isConfirmation(content string) bool {
	content = strings.TrimSpace(strings.ToLower(content))

	// Real confirmations are brief.
	if len(content) > 60 {
		return false
	}

	// Strip trailing punctuation for matching.
	cleaned := strings.TrimRight(content, "!.?, ")

	for _, pattern := range confirmationPatterns {
		if cleaned == pattern {
			return true
		}
	}
	return false
}

// lastUserMessage returns the content of the last user message in the request,
// with <system-reminder> tags stripped.
func lastUserMessage(req *types.ChatCompletionRequest) string {
	if req == nil {
		return ""
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if strings.ToLower(req.Messages[i].Role) == "user" {
			return strings.TrimSpace(stripSystemReminders(req.Messages[i].ContentString()))
		}
	}
	return ""
}

// assistantContextScore analyses the last assistant message before the final
// user message to determine the conversation's complexity context. This is
// used when the user sends a short confirmation like "do it" — the complexity
// of the task lives in the assistant's preceding proposal, not in the user's
// two-word reply.
func assistantContextScore(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}

	// Walk backwards: skip past the last user message, then find the
	// assistant message immediately before it.
	foundUser := false
	var assistantContent string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		role := strings.ToLower(req.Messages[i].Role)
		if !foundUser && role == "user" {
			foundUser = true
			continue
		}
		if foundUser && role == "assistant" {
			assistantContent = req.Messages[i].ContentString()
			break
		}
	}

	if assistantContent == "" {
		return 0.0
	}

	lower := strings.ToLower(assistantContent)
	var score float64

	// Code blocks indicate implementation context.
	codeBlocks := strings.Count(lower, "```")
	if codeBlocks >= 4 {
		score = math.Max(score, 0.85)
	} else if codeBlocks >= 2 {
		score = math.Max(score, 0.65)
	}

	// Implementation / architecture language.
	if containsAny(lower, "implement", "refactor", "architect", "design pattern",
		"migration", "restructure", "overhaul") {
		score = math.Max(score, 0.75)
	}

	// Multi-step plans.
	if containsAny(lower, "step 1", "1.") &&
		containsAny(lower, "step 2", "2.", "then") {
		score = math.Max(score, 0.70)
	}

	// Debugging context.
	if containsAny(lower, "the bug", "the error", "root cause", "stack trace") {
		score = math.Max(score, 0.55)
	}

	// Long assistant responses suggest complex context.
	if len(assistantContent) > 3000 {
		score = math.Max(score, 0.60)
	} else if len(assistantContent) > 1000 {
		score = math.Max(score, 0.40)
	}

	return score
}

// userHintsSignal incorporates explicit user preferences into the score.
// Weight: 0.15
func userHintsSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil || req.AIONPreferences == nil {
		return 0.5
	}

	prefs := req.AIONPreferences

	if prefs.PreferredTier != nil {
		tier := *prefs.PreferredTier
		return float64(tier-1) / 2.0
	}
	if prefs.PreferQuality {
		return 0.8
	}
	if prefs.PreferSpeed {
		return 0.1
	}
	return 0.5
}
