package classifier

import "github.com/ShubhamDX/aion/internal/types"

// TierFromScore maps a weighted score to a Tier given the configured thresholds.
// Scores below t1Threshold are Tier1, below t2Threshold are Tier2, otherwise Tier3.
func TierFromScore(score float64, t1Threshold, t2Threshold float64) types.Tier {
	if score < t1Threshold {
		return types.Tier1
	}
	if score < t2Threshold {
		return types.Tier2
	}
	return types.Tier3
}
