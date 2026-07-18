package proxy

import "context"

// Ingress protocol identity. The proxy's core lifecycle (ChatCompletion) is
// reused by more than one wire endpoint: /v1/chat/completions is native
// "openai_chat", while the OpenAI Responses endpoint translates to the same
// lifecycle but is really "openai_responses", and the Anthropic ingress is
// "anthropic". The ResponseAction hook must be told the ACTUAL ingress so a
// policy can distinguish the contracts, so the true protocol is carried
// explicitly through the request context rather than hard-coded at the hook call
// site.
const (
	protocolOpenAIChat      = "openai_chat"
	protocolOpenAIResponses = "openai_responses"
	protocolAnthropic       = "anthropic"
)

type ingressProtocolKey struct{}

// withIngressProtocol returns a context that carries the true ingress protocol.
func withIngressProtocol(ctx context.Context, protocol string) context.Context {
	return context.WithValue(ctx, ingressProtocolKey{}, protocol)
}

// ingressProtocol reads the ingress protocol from the context, defaulting to
// "openai_chat" (the native /v1/chat/completions contract) when unset.
func ingressProtocol(ctx context.Context) string {
	if v, ok := ctx.Value(ingressProtocolKey{}).(string); ok && v != "" {
		return v
	}
	return protocolOpenAIChat
}
