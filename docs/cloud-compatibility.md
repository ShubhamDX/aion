# Cloud compatibility and offline tests

Reviewed September 19, 2026. An adapter contract test checks AION's wire format.
It does not establish that a cloud account can serve a particular model.

## Available paths

| Cloud path | AION configuration | Contract and limits |
|---|---|---|
| Claude on Vertex | `providers.vertex`, `project_id`, `region`, static bearer in `api_key` | Native `rawPredict` and `streamRawPredict`. Supports named and required tools, text cache checkpoints and native usage translation. This slot is not a Gemini `generateContent` adapter. |
| Gemini on Vertex | `providers.gemini`, custom `base_url`, static OAuth token in `api_key` | OpenAI-compatible Chat Completions. AION does not refresh Google credentials automatically. |
| Azure OpenAI v1 | `providers.openai`, custom `base_url`, static API key or bearer token in `api_key` | OpenAI-compatible Chat Completions. `model` is the Azure deployment name. No dedicated Azure slot, legacy dated API transport or automatic Entra token refresh. |
| Gemini Developer API | `providers.gemini` | OpenAI-compatible endpoint. Not the native Gemini REST schema. |

Set the base URL before AION appends `/chat/completions`:

```text
Azure OpenAI v1:
https://RESOURCE.openai.azure.com/openai/v1

Gemini on Vertex:
https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/LOCATION/endpoints/openapi
```

Each provider slot has one endpoint. An Azure endpoint configured under `openai`
still has AION provider identity `openai`; a Vertex endpoint under `gemini`
still has identity `gemini`. This is not a multi-account cloud registry.
Set the exact model or deployment IDs, prices and limits from the customer
account. Compatible protocol support does not imply every model supports every
tool, schema, cache or reasoning option.

Vertex Claude preserves forced tool choices without silently weakening them.
Unsupported response-format requirements fail before dispatch. A provider
rejection is returned without an unconstrained retry. Text cache checkpoints
are retained on native Claude requests and removed from Gemini-compatible
requests, where Anthropic's `cache_control` field is not a portable contract.
This does not disable any provider-managed automatic caching.

Compatible responses retain `message.refusal` and streaming `delta.refusal`.
Native Claude refusals retain the terminal `refusal` finish reason. The
OpenAI-shaped response does not expose Claude's native `stop_details` extension.
A refusal does not become a successful tool submission.

## Test without inference calls

Run from the OSS repository:

```sh
GOWORK=off go test ./internal/provider ./pkg/app -run '^TestCloudOffline' -count=1 -v
```

These tests use synthetic payloads, dummy credentials and local HTTP servers.
Adapter transports accept only the configured test server. Gateway tests reject
non-loopback destinations. No keychain, cloud credentials, SDK login, model
download or provider diagnostic is required.

The tests cover endpoint paths, bearer headers, deployment selection, tool
history, forced choices, streaming, refusals, exact integer arguments, caller
immutability and native cache-usage partitions. They exercise the gateway's
configuration and HTTP dispatch as well as each adapter.

For an additional operating-system boundary on macOS, after Go dependencies
are cached:

```sh
GOWORK=off sandbox-exec \
  -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))' \
  go test ./internal/provider ./pkg/app -run '^TestCloudOffline' -count=1 -v
```

The full OSS suite also passes with that network restriction. Go dependency
downloads require a separate preparation step if the local module cache is
empty. Do not replace an offline test with the dashboard's provider test:
that test sends a one-token inference and may be billed.

## What remains unverified

Local fixtures cannot validate IAM, token expiry handling by the cloud, quotas,
deployment availability, real model output, provider latency, cache hits or
billed cost. They also cannot establish model-specific schema acceptance.
Only a separately authorized live qualification can measure those properties.

The reviewed Google emulator command list does not list a managed Vertex
foundation-model emulator. Microsoft's Azure SDK tools offer mocking and
record/playback; API Management can synthesize responses with `mock-response`.
Those tools test contracts, not cloud model behavior. A provider playground
or trial account still invokes the provider when it generates output.

## Official references

- [Claude on Vertex](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude)
- [Vertex OpenAI-compatible API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate/openai/overview)
- [Vertex authentication and credential refresh](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate/openai/auth-and-credentials)
- [Claude prompt caching on Vertex](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/prompt-caching)
- [Azure OpenAI v1 lifecycle](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)
- [Google Cloud emulators](https://docs.cloud.google.com/sdk/gcloud/reference/emulators)
- [Azure SDK unit testing and mocking](https://learn.microsoft.com/en-us/dotnet/azure/sdk/unit-testing-mocking)
- [Azure SDK test proxy](https://github.com/Azure/azure-sdk-tools/blob/main/tools/test-proxy/Azure.Sdk.Tools.TestProxy/README.md)
- [API Management mock-response policy](https://learn.microsoft.com/en-us/azure/api-management/mock-response-policy)
