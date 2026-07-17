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
type PostRouteInput = pkgtypes.PostRouteInput
type ContextCompressionResult = pkgtypes.ContextCompressionResult
type OutputControlInput = pkgtypes.OutputControlInput
type OutputControlResult = pkgtypes.OutputControlResult
type SessionMaterial = pkgtypes.SessionMaterial
type SessionSource = pkgtypes.SessionSource
type ProposedToolCall = pkgtypes.ProposedToolCall
type ResponseActionInput = pkgtypes.ResponseActionInput
type ResponseActionVerdict = pkgtypes.ResponseActionVerdict
type ResponseActionCallDecision = pkgtypes.ResponseActionCallDecision
type ResponseActionDecision = pkgtypes.ResponseActionDecision

const (
	VerdictAllow = pkgtypes.VerdictAllow
	VerdictRoute = pkgtypes.VerdictRoute
	VerdictBlock = pkgtypes.VerdictBlock
	VerdictHold  = pkgtypes.VerdictHold

	ActionAllow = pkgtypes.ActionAllow
	ActionBlock = pkgtypes.ActionBlock
	ActionHold  = pkgtypes.ActionHold

	MaxBufferedToolCallArgsBytes = pkgtypes.MaxBufferedToolCallArgsBytes

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
	ScrubSessionID             = pkgtypes.ScrubSessionID
	ResponseContentStrings     = pkgtypes.ResponseContentStrings
	ArgsDigestHex              = pkgtypes.ArgsDigestHex
)
