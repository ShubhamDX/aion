# Cloud compatibility and offline tests

Reviewed September 19, 2026. An adapter contract test checks AION's wire format.
It does not establish that a cloud account can serve a particular model.

## Available paths

| Cloud path | AION configuration | Contract and limits |
|---|---|---|
| Claude on Vertex | `providers.vertex`, `project_id`, `region`, static bearer or `credential_mode: google_adc` | Native `rawPredict` and `streamRawPredict`. Supports named and required tools, text cache checkpoints and native usage translation. This slot is not a Gemini `generateContent` adapter. |
| Gemini on Vertex | `providers.gemini`, custom `base_url`, static OAuth token or `credential_mode: google_adc` | OpenAI-compatible Chat Completions. The Google SDK refreshes ADC tokens in managed mode. |
| Azure OpenAI v1 | `providers.openai`, custom `base_url`, static key or `credential_mode: azure_entra` | OpenAI-compatible Chat Completions. `model` is the Azure deployment name. The Azure SDK supplies Entra tokens in managed mode. No dedicated Azure slot or legacy dated API transport. |
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

## Managed credential refresh

Source implementation, not a live cloud qualification claim. The selected
release must include this change before these modes can be used in a bundle.

Set `credential_mode: google_adc` on `vertex` or Vertex-compatible `gemini`.
The official Google Go auth library discovers Application Default Credentials,
requests the cloud-platform scope and refreshes cached tokens before expiry.
ADC quota project headers are preserved. The configured `project_id` remains
the inference project; credential discovery does not replace that value.

Set `credential_mode: azure_entra` on the Azure v1 `openai` slot. The official
Azure Go identity library uses `DefaultAzureCredential` and requests
`https://ai.azure.com/.default`. Existing environment, workload identity,
managed identity or supported CLI credentials can supply the identity.
Production deployments should select their intended credential using
`AZURE_TOKEN_CREDENTIALS`, for example `WorkloadIdentityCredential`,
`ManagedIdentityCredential` or `EnvironmentCredential`. SDK caching behavior
depends on the selected credential; CLI credentials may acquire a token on each
request. No interactive login is launched by AION.

```yaml
providers:
  vertex:
    credential_mode: google_adc
    project_id: YOUR_PROJECT
    region: us-east5
    # Add exact enabled models and verified prices to models.
  openai:
    credential_mode: azure_entra
    base_url: https://YOUR_RESOURCE.openai.azure.com/openai/v1
    # Add exact Azure deployment names and verified prices to models.
```

Use trusted SDK credential configuration mounted by the operator. Do not accept
ADC files or identity endpoints from request payloads. Keep credential files
outside source control. AION does not mint a new permanent cloud key.

Managed mode rejects a simultaneous `api_key`, unrelated endpoints, HTTP and
redirects. Only canonical public Google AI Platform and Azure v1 hosts are
accepted. Native Vertex requires a regional origin, or an explicit
`https://aiplatform.googleapis.com` base URL when using `region: global`.
Sovereign clouds and proxy hostnames require a separately reviewed transport.
Static modes retain their existing endpoint and key behavior and do not gain
automatic refresh.

Normal and streaming requests use the same credential transport. Failed token
refresh stops inference with a sanitized credential error. AION does not retry
an inference after a 401 or switch to a static key. Identity SDKs may retry
token acquisition within the request context. The caller's deadline also
bounds credential acquisition. Google token HTTP requests have a 15-second
client timeout.

Compatible streaming callers can set `stream_options.include_usage: true`.
Native Claude usage arrives in separate input/output chunks, so cost consumers
must merge cumulative fields instead of replacing the previous usage object.
Missing usage is unknown cost, not zero cost.

## Test without inference calls

Run from the OSS repository:

```sh
GOWORK=off go test ./internal/provider ./pkg/app -run '^(TestCloudOffline|TestCloudAuth|TestCloudCompatible)' -count=1 -v
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
  go test ./internal/provider ./pkg/app -run '^(TestCloudOffline|TestCloudAuth|TestCloudCompatible)' -count=1 -v
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

- [Google Go ADC library](https://pkg.go.dev/cloud.google.com/go/auth/credentials)
- [Azure Go identity library](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity)

- [Claude on Vertex](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude)
- [Vertex OpenAI-compatible API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate/openai/overview)
- [Vertex authentication and credential refresh](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate/openai/auth-and-credentials)
- [Claude prompt caching on Vertex](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/prompt-caching)
- [Azure OpenAI v1 lifecycle](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)
- [Google Cloud emulators](https://docs.cloud.google.com/sdk/gcloud/reference/emulators)
- [Azure SDK unit testing and mocking](https://learn.microsoft.com/en-us/dotnet/azure/sdk/unit-testing-mocking)
- [Azure SDK test proxy](https://github.com/Azure/azure-sdk-tools/blob/main/tools/test-proxy/Azure.Sdk.Tools.TestProxy/README.md)
- [API Management mock-response policy](https://learn.microsoft.com/en-us/azure/api-management/mock-response-policy)
