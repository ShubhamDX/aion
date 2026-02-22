package classifier

import (
	pkgtypes "github.com/ShubhamDX/aion/pkg/types"
)

// Classifier routes chat completion requests to complexity tiers.
type Classifier interface {
	Classify(req *pkgtypes.ChatCompletionRequest) (pkgtypes.Tier, float64, map[string]float64)
}
