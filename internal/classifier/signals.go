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
