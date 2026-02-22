package classifier

import (
	pkgtypes "github.com/ShubhamDX/aion/pkg/types"
)

// TierFromScore maps a weighted score to a Tier given the configured thresholds.
// Scores below t1Threshold are Tier1, below t2Threshold are Tier2, otherwise Tier3.
func TierFromScore(score float64, t1Threshold, t2Threshold float64) pkgtypes.Tier {
	if score < t1Threshold {
		return pkgtypes.Tier1
	}
	if score < t2Threshold {
		return pkgtypes.Tier2
	}
	return pkgtypes.Tier3
}
