package proxy

import (
	"github.com/ShubhamDX/aion/internal/apikey"
	"github.com/ShubhamDX/aion/internal/router"
	"github.com/ShubhamDX/aion/internal/types"
)

func (h *Handler) applyOutputControl(
	req *types.ChatCompletionRequest,
	requestID string,
	keyInfo *apikey.KeyInfo,
	requestedModel string,
	selectedModel *router.ModelOption,
	tier types.Tier,
) {
	if req == nil || selectedModel == nil || h.hooks == nil || h.hooks.ApplyOutputControl == nil {
		return
	}
	res := h.hooks.ApplyOutputControl(types.OutputControlInput{
		PostRouteInput: types.PostRouteInput{
			RequestID:      requestID,
			PrincipalID:    keyIDFromInfo(keyInfo),
			RequestedModel: requestedModel,
			RoutedProvider: selectedModel.Provider,
			RoutedModel:    selectedModel.ID,
			Tier:           tier,
			RequestDigest:  types.RequestContentDigest(req),
		},
		Request: req,
	})
	if res == nil {
		return
	}
	if res.Messages != nil {
		req.Messages = res.Messages
	}
	if res.MaxTokens != nil {
		cap := *res.MaxTokens
		req.MaxTokens = &cap
	}
}
