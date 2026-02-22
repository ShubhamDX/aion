package config

// ClassifierConfig holds thresholds for the complexity classifier.
type ClassifierConfig struct {
	Tier1Threshold  float64 `yaml:"tier1_threshold"`
	Tier2Threshold  float64 `yaml:"tier2_threshold"`
	IntentModelPath string  `yaml:"intent_model_path,omitempty"`
}
