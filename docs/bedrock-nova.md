# Native Nova transport

Configured Amazon Nova model IDs use Bedrock Converse for non-streaming calls and
ConverseStream for streaming calls. The provider reuses the configured bearer or
AWS SDK signing credentials. It does not create credentials.

Supported inputs are text, system/developer instructions, tool definitions,
assistant tool calls and tool results. Consecutive messages with the same role
are merged in source order. Unsupported content types fail explicitly instead of
being flattened to text. Tool calls preserve their IDs, names and JSON arguments.

Explicit text cache controls map to system/user `cachePoint` blocks. The accepted
cache control is `type: ephemeral` with omitted TTL or `5m`. Assistant/tool cache
checkpoints and other TTLs fail explicitly. Actual cache eligibility and hits
depend on the configured model and provider; a checkpoint is not a promised hit.

Nova's duplicate cache counter names describe the same tokens. The adapter counts
each class once and requires total usage to equal uncached input plus cache reads,
cache writes and output. Conflicting aliases or inconsistent totals fail instead
of returning invented usage. Streaming metadata supplies the same partition.

The tested Nova 2 Lite endpoint rejected `outputConfig`. The tested Claude Opus 5
and Sonnet 5 endpoints rejected `output_config.format`. Non-text caller
`response_format` and mandatory native schema settings fail explicitly on these
transports. Mantle retains its existing native schema path.

Session fingerprints normalize supported text-only checkpoint metadata. Moving a
checkpoint does not erase conversation continuity. Source changes and unknown
content fields still invalidate the prefix. Request evidence digests retain the
original request fields.

Validation includes local HTTP and event-stream tests for text, tool history,
authentication, cache counters, rejected contracts and malformed frame lengths.
The opt-in live stream test makes one bounded paid call using the existing token:

```sh
AION_RUN_NOVA_LIVE=1 go test ./internal/provider -run TestNovaLiveStream -v
```

Live verification on 17 September 2026 covered native non-streaming requests with
tool history, a system cache-write checkpoint and streamed text with usage. This
is not a certification of every Nova model or multimodal input.
