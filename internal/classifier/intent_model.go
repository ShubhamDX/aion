package classifier

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
)

// IntentModel holds the deserialized TF-IDF + Logistic Regression model
// exported from the Python training notebook. All inference runs in pure Go
// with no external dependencies.
type IntentModel struct {
	Vocabulary      map[string]int     `json:"vocabulary"`
	IDFWeights      []float64          `json:"idf_weights"`
	ModelWeights    [][]float64        `json:"model_weights"`    // [n_classes][n_features]
	ModelIntercepts []float64          `json:"model_intercepts"` // [n_classes]
	Categories      []string           `json:"categories"`
	CategoryScores  map[string]float64 `json:"category_scores"`
	Config          modelConfig        `json:"config"`
}

type modelConfig struct {
	SublinearTF  bool   `json:"sublinear_tf"`
	Norm         string `json:"norm"`
	TokenPattern string `json:"token_pattern"`
}

// tokenRegexp matches the default sklearn token pattern: 2+ word characters
// bounded by word boundaries. Compiled once at package init.
var tokenRegexp = regexp.MustCompile(`\b\w\w+\b`)

// LoadIntentModel reads and validates an intent model from a JSON file.
func LoadIntentModel(path string) (*IntentModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("intent model: read: %w", err)
	}
	return LoadIntentModelFromBytes(data)
}

// LoadIntentModelFromBytes parses and validates an intent model from raw
// JSON bytes. This is used both by LoadIntentModel (file-based) and by the
// embedded default model loader.
func LoadIntentModelFromBytes(data []byte) (*IntentModel, error) {
	var m IntentModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("intent model: parse: %w", err)
	}

	if err := m.validate(); err != nil {
		return nil, fmt.Errorf("intent model: validate: %w", err)
	}

	return &m, nil
}

func (m *IntentModel) validate() error {
	if len(m.Vocabulary) == 0 {
		return fmt.Errorf("empty vocabulary")
	}
	if len(m.IDFWeights) == 0 {
		return fmt.Errorf("empty IDF weights")
	}
	if len(m.ModelWeights) == 0 {
		return fmt.Errorf("empty model weights")
	}
	if len(m.ModelWeights) != len(m.ModelIntercepts) {
		return fmt.Errorf("weight/intercept dimension mismatch: %d != %d",
			len(m.ModelWeights), len(m.ModelIntercepts))
	}
	nFeatures := len(m.IDFWeights)
	for i, row := range m.ModelWeights {
		if len(row) != nFeatures {
			return fmt.Errorf("weight row %d has %d features, expected %d", i, len(row), nFeatures)
		}
	}
	if len(m.Categories) != len(m.ModelWeights) {
		return fmt.Errorf("categories/weights dimension mismatch: %d != %d",
			len(m.Categories), len(m.ModelWeights))
	}
	return nil
}

// Predict returns a complexity score for the given text by running TF-IDF
// vectorization and logistic regression inference, then computing a
// probability-weighted category score.
func (m *IntentModel) Predict(text string) float64 {
	tfidf := m.vectorize(text)
	if tfidf == nil {
		return 0.0
	}

	probabilities := m.classify(tfidf)
	return m.weightedScore(probabilities)
}

// vectorize converts raw text into a TF-IDF feature vector matching the
// sklearn pipeline's behavior.
func (m *IntentModel) vectorize(text string) []float64 {
	lower := strings.ToLower(text)
	tokens := tokenRegexp.FindAllString(lower, -1)
	if len(tokens) == 0 {
		return nil
	}

	nFeatures := len(m.IDFWeights)
	vec := make([]float64, nFeatures)

	// Count term frequencies.
	tf := make(map[int]float64, len(tokens))
	for _, tok := range tokens {
		if idx, ok := m.Vocabulary[tok]; ok {
			tf[idx]++
		}
	}

	if len(tf) == 0 {
		return nil
	}

	// Apply sublinear TF scaling and IDF.
	for idx, count := range tf {
		v := count
		if m.Config.SublinearTF {
			v = 1.0 + math.Log(count)
		}
		vec[idx] = v * m.IDFWeights[idx]
	}

	// L2 normalization.
	if m.Config.Norm == "l2" {
		var norm float64
		for _, v := range vec {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for i := range vec {
				vec[i] /= norm
			}
		}
	}

	return vec
}

// classify runs logistic regression: logits = W·x + b, then softmax.
func (m *IntentModel) classify(tfidf []float64) []float64 {
	nClasses := len(m.ModelWeights)
	logits := make([]float64, nClasses)

	for c := 0; c < nClasses; c++ {
		sum := m.ModelIntercepts[c]
		weights := m.ModelWeights[c]
		for i, v := range tfidf {
			sum += weights[i] * v
		}
		logits[c] = sum
	}

	return softmax(logits)
}

// weightedScore computes sum(probability_i * category_score_i) across all
// classes, producing a continuous complexity score.
func (m *IntentModel) weightedScore(probabilities []float64) float64 {
	var score float64
	for i, cat := range m.Categories {
		if s, ok := m.CategoryScores[cat]; ok {
			score += probabilities[i] * s
		}
	}
	return math.Min(1.0, score)
}

// softmax converts logits to a probability distribution.
func softmax(logits []float64) []float64 {
	// Subtract max for numerical stability.
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}

	probs := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		probs[i] = math.Exp(v - maxLogit)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}
