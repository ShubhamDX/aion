package classifier

import (
	"log"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
	"github.com/ShubhamDX/aion/models"
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
//
// Intent model resolution order:
//  1. Custom model at IntentModelPath (user-trained on their own data)
//  2. Embedded default model (ships with the binary, trained on anonymized data)
//  3. Pattern-matching heuristics (zero-dependency fallback)
func New(cfg config.ClassifierConfig) *Classifier {
	var intentModel *IntentModel

	// 1. Try custom model from config path.
	if cfg.IntentModelPath != "" {
		m, err := LoadIntentModel(cfg.IntentModelPath)
		if err != nil {
			log.Printf("classifier: custom intent model not loaded (%v), trying default", err)
		} else {
			log.Printf("classifier: loaded custom intent model from %s (%d vocab, %d classes)",
				cfg.IntentModelPath, len(m.Vocabulary), len(m.Categories))
			intentModel = m
		}
	}

	// 2. Fall back to embedded default model.
	if intentModel == nil {
		m, err := LoadIntentModelFromBytes(models.DefaultIntentModelJSON)
		if err != nil {
			log.Printf("classifier: embedded default model failed (%v), using pattern fallback", err)
		} else {
			log.Printf("classifier: using embedded default intent model (%d vocab, %d classes)",
				len(m.Vocabulary), len(m.Categories))
			intentModel = m
		}
	}

	return &Classifier{
		signals: []WeightedSignal{
			{"token_volume", 0.10, tokenVolumeSignal},
			{"message_count", 0.05, messageCountSignal},
			{"system_prompt", 0.05, systemPromptSignal},
			{"tool_presence", 0.05, toolPresenceSignal},
			{"content_keywords", 0.25, contentKeywordsSignal},
			{"intent", 0.35, makeIntentSignal(intentModel)},
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
