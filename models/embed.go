// Package models provides the default embedded intent classifier model.
package models

import _ "embed"

// DefaultIntentModelJSON holds the pre-trained default intent classifier
// model. This model is trained on anonymized + synthetic data and is safe
// for public distribution. Users can override it by setting
// intent_model_path in their AION configuration.
//
//go:embed intent_classifier_default.json
var DefaultIntentModelJSON []byte
