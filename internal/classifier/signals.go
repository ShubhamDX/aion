package classifier

import (
	"math"
	"strings"

	"github.com/ShubhamDX/aion/internal/types"
)

// SignalFunc extracts a complexity signal from a request, returning 0.0-1.0.
type SignalFunc func(req *types.ChatCompletionRequest) float64

// tokenVolumeSignal estimates token count as total content length / 4
// and scores it against a 4000-token reference.
// Weight: 0.20
func tokenVolumeSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}
	totalChars := 0
	for _, msg := range req.Messages {
		totalChars += len(msg.Content)
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

// systemPromptSignal evaluates the system message for length and complexity keywords.
// Weight: 0.15
func systemPromptSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}

	var systemContent string
	for _, msg := range req.Messages {
		if strings.ToLower(msg.Role) == "system" {
			systemContent = msg.Content
			break
		}
	}

	if systemContent == "" {
		return 0.0
	}

	lower := strings.ToLower(systemContent)
	lengthScore := math.Min(1.0, float64(len(systemContent))/2000.0)

	keywords := []string{"analyze", "reason", "step-by-step", "chain of thought", "expert", "detailed"}
	matchCount := 0
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			matchCount++
		}
	}
	keywordBonus := math.Min(1.0, float64(matchCount)/float64(len(keywords)))

	score := lengthScore*0.6 + keywordBonus*0.4
	return math.Min(1.0, score)
}

// toolPresenceSignal scores the number of tools against a 5-tool reference.
// Weight: 0.15
func toolPresenceSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}
	return math.Min(1.0, float64(len(req.Tools))/5.0)
}

// contentKeywordsSignal checks all user messages for complexity indicators.
// Weight: 0.25
func contentKeywordsSignal(req *types.ChatCompletionRequest) float64 {
	if req == nil {
		return 0.0
	}

	// Concatenate all user messages for keyword analysis.
	var sb strings.Builder
	for _, msg := range req.Messages {
		if strings.ToLower(msg.Role) == "user" {
			sb.WriteString(msg.Content)
			sb.WriteByte(' ')
		}
	}

	content := sb.String()
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
