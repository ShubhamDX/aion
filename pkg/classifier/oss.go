package classifier

import (
	pkgconfig "github.com/ShubhamDX/aion/pkg/config"

	// Import internal classifier — allowed because this is the same module.
	"github.com/ShubhamDX/aion/internal/classifier"
	"github.com/ShubhamDX/aion/internal/config"
)

// NewOSS creates the open-source TF-IDF classifier. External modules (e.g.
// aion-enterprise) use this factory to construct the OSS classifier without
// importing internal packages directly.
func NewOSS(cfg pkgconfig.ClassifierConfig) Classifier {
	return classifier.New(config.ClassifierConfig(cfg))
}
