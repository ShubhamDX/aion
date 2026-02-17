package classifier

import (
	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

// WeightedSignal pairs a named signal extractor with its weight in the overall score.
type WeightedSignal struct {
	Name   string
	Weight float64
	Fn     SignalFunc
}

// Classifier evaluates chat completion requests and assigns them a complexity tier
// based on a weighted combination of heuristic signals.
type Classifier struct {
	signals     []WeightedSignal
	t1Threshold float64
	t2Threshold float64
}

// New creates a Classifier with the default signal set and the tier thresholds
// from the supplied configuration.
func New(cfg config.ClassifierConfig) *Classifier {
	return &Classifier{
		signals: []WeightedSignal{
			{"token_volume", 0.20, tokenVolumeSignal},
			{"message_count", 0.10, messageCountSignal},
			{"system_prompt", 0.15, systemPromptSignal},
			{"tool_presence", 0.15, toolPresenceSignal},
			{"content_keywords", 0.25, contentKeywordsSignal},
			{"user_hints", 0.15, userHintsSignal},
		},
		t1Threshold: cfg.Tier1Threshold,
		t2Threshold: cfg.Tier2Threshold,
	}
}

// Classify analyses the request and returns the assigned tier, the raw
// weighted score, and a breakdown of individual signal values.
func (c *Classifier) Classify(req *types.ChatCompletionRequest) (types.Tier, float64, map[string]float64) {
	signals := make(map[string]float64, len(c.signals))
	var totalScore float64
	for _, ws := range c.signals {
		val := ws.Fn(req)
		signals[ws.Name] = val
		totalScore += val * ws.Weight
	}
	tier := TierFromScore(totalScore, c.t1Threshold, c.t2Threshold)
	return tier, totalScore, signals
}
