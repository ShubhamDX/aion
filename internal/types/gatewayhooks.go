package types

import (
	pkgtypes "github.com/ShubhamDX/aion/pkg/types"
)

// Gateway hook aliases so internal/proxy can reference the stable pkg/types
// extension surface through the internal types package it already imports.
type GatewayHooks = pkgtypes.GatewayHooks
type PreRequestInput = pkgtypes.PreRequestInput
type PreRequestDecision = pkgtypes.PreRequestDecision
type PreRequestVerdict = pkgtypes.PreRequestVerdict
type PostResponseInput = pkgtypes.PostResponseInput

const (
	VerdictAllow = pkgtypes.VerdictAllow
	VerdictRoute = pkgtypes.VerdictRoute
	VerdictBlock = pkgtypes.VerdictBlock
	VerdictHold  = pkgtypes.VerdictHold
)

// Content-digest helpers (re-exported for internal/proxy).
var (
	RequestContentDigest  = pkgtypes.RequestContentDigest
	ResponseContentDigest = pkgtypes.ResponseContentDigest
)
