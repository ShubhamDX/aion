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
type SessionMaterial = pkgtypes.SessionMaterial
type SessionSource = pkgtypes.SessionSource

const (
	VerdictAllow = pkgtypes.VerdictAllow
	VerdictRoute = pkgtypes.VerdictRoute
	VerdictBlock = pkgtypes.VerdictBlock
	VerdictHold  = pkgtypes.VerdictHold

	SessionSourceHeader  = pkgtypes.SessionSourceHeader
	SessionSourceBody    = pkgtypes.SessionSourceBody
	SessionSourceDerived = pkgtypes.SessionSourceDerived
	SessionSourceNone    = pkgtypes.SessionSourceNone
)

// Content-digest + session-material helpers (re-exported for internal/proxy).
var (
	RequestContentDigest       = pkgtypes.RequestContentDigest
	ResponseContentDigest      = pkgtypes.ResponseContentDigest
	SessionMaterialFromRequest = pkgtypes.SessionMaterialFromRequest
	NextCachePrefixMaterial    = pkgtypes.NextCachePrefixMaterial
)
