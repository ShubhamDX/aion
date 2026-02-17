# AION -- Intelligent LLM Cost Router

**Same quality. 40-70% cheaper.**

AION analyzes the complexity of your LLM requests and routes them to the cheapest model that can handle it. Simple questions go to cheap models. Complex reasoning goes to capable ones. You save money without sacrificing quality.

> AION is NOT another OpenRouter/LiteLLM. Those are dumb pipes -- you pick a model, they forward. AION is different: you send a request without specifying a model, and AION figures out the right one.

## Quick Start

### 1. Get AION

**From source:**

```bash
git clone https://github.com/ShubhamDX/aion.git
cd aion
make build
```

**With Docker:**

```bash
docker pull aion:latest
# or build locally
make docker-build
```

### 2. Configure

```bash
cp configs/aion.example.yaml configs/aion.yaml
```

### 3. Set API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Run

```bash
# Binary
./bin/aion -config configs/aion.yaml

# Docker
make docker-run
```

### 5. Point your OpenAI SDK at AION

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-aion-dev-key-change-me",
)

# Don't specify a model -- AION picks the cheapest one that works
response = client.chat.completions.create(
    model="",  # leave empty for auto-routing
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(response.choices[0].message.content)
# Response headers include X-AION-Model, X-AION-Tier, X-AION-Cost-USD
```

That's it. Your existing code works unchanged -- just swap the base URL.

## How It Works

AION uses a heuristic classifier that scores each request's complexity in under 1ms. The classifier evaluates 6 signals:

| Signal | What it measures | Weight |
|---|---|---|
| Message length | Total character count across all messages | Weighted |
| Vocabulary complexity | Ratio of unique words to total words | Weighted |
| Conversation turns | Number of back-and-forth messages | Weighted |
| System prompt presence | Whether a system prompt is provided | Weighted |
| Tool usage | Whether tools/functions are requested | Weighted |
| Explicit preference | `aion_preferences` field in the request | Override |

The classifier produces a score between 0.0 and 1.0:

- **Score < 0.35** -- Tier 1 (simple): Routed to cheapest models (gpt-4o-mini, claude-haiku-3-5)
- **Score 0.35 - 0.70** -- Tier 2 (moderate): Routed to mid-range models (gpt-4o, claude-sonnet-4)
- **Score > 0.70** -- Tier 3 (complex): Routed to most capable models (o1, claude-opus-4)

Within each tier, AION picks the cheapest available model. If a tier has no healthy models, it falls back to an adjacent tier.

## Architecture

```
                    Incoming Request
                          |
                    +-----v------+
                    |   Auth &   |
                    |   Budget   |
                    +-----+------+
                          |
                    +-----v------+
                    | Classifier |  <1ms overhead
                    |  (6 signals)|
                    +-----+------+
                          |
               +----------+----------+
               |          |          |
          Tier 1     Tier 2     Tier 3
          (simple)   (moderate) (complex)
               |          |          |
            +--v--+    +--v--+   +--v--+
            |Route|    |Route|   |Route|
            +--+--+    +--+--+  +--+--+
               |          |          |
          +----v---+ +----v---+ +---v----+
          |gpt-4o  | | gpt-4o | |   o1   |
          | -mini  | |claude- | |claude- |
          |haiku   | |sonnet  | | opus   |
          +--------+ +--------+ +--------+
                          |
                    +-----v------+
                    | Telemetry  |
                    |  (async)   |
                    +------------+
```

Key design decisions:

- **OpenAI-compatible API**: Drop-in replacement -- change only the base URL
- **Zero external dependencies at runtime**: Classifier is pure Go heuristics, no ML model needed
- **Async telemetry**: Request logging is batched and written to SQLite without blocking responses
- **Graceful degradation**: If a provider is down, AION falls back to the next cheapest option

## API

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint. Supports both streaming (`stream: true`) and non-streaming requests.

**Request:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-aion-dev-key-change-me" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "stream": false
  }'
```

**With routing hints:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-aion-dev-key-change-me" \
  -d '{
    "messages": [{"role": "user", "content": "Write a compiler"}],
    "aion_preferences": {
      "preferred_tier": 3,
      "max_cost_usd": 0.50,
      "prefer_quality": true
    }
  }'
```

**Response headers:**

| Header | Description |
|---|---|
| `X-AION-Model` | The model that handled the request |
| `X-AION-Provider` | The provider (openai, anthropic, openrouter) |
| `X-AION-Tier` | Complexity tier assigned (1, 2, or 3) |
| `X-AION-Score` | Raw classifier score (0.0 - 1.0) |
| `X-AION-Cost-USD` | Estimated cost for this request |
| `X-AION-Request-ID` | Unique request identifier |

### `GET /v1/models`

Lists all available models across configured providers.

### `GET /health`

Returns `200 OK` with system status.

### SDK Examples

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-aion-dev-key-change-me",
)

# Auto-routed -- AION picks the model
response = client.chat.completions.create(
    model="",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
    ],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="",
    messages=[{"role": "user", "content": "Write a haiku about Go"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**JavaScript (OpenAI SDK):**

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "sk-aion-dev-key-change-me",
});

const response = await client.chat.completions.create({
  model: "",
  messages: [{ role: "user", content: "What's the capital of France?" }],
});
console.log(response.choices[0].message.content);
```

**curl (streaming):**

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-aion-dev-key-change-me" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Configuration

Configuration lives in `configs/aion.yaml`. See [`configs/aion.example.yaml`](configs/aion.example.yaml) for a fully documented example.

### Key sections

**server** -- HTTP port and timeouts. Defaults to port 8080.

**auth** -- Enable/disable API key authentication. Each key can have daily and monthly budget limits.

**providers** -- Configure one or more LLM providers (OpenAI, Anthropic, OpenRouter). Each provider lists its available models with tier assignments and pricing.

**routing** -- Controls the routing strategy (`cheapest` or `fallback`) and classifier thresholds. The thresholds determine the score boundaries between tiers.

**telemetry** -- SQLite-based request logging. Configurable batch size and flush interval for async writes.

### Environment variables

API keys should be set as environment variables rather than hardcoded in the config file:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
```

The config file supports `${VAR}` and `$VAR` syntax for environment variable expansion.

## Building from Source

```bash
# Build the binary
make build

# Run tests
make test

# Run tests with coverage report
make test-cover

# Run benchmarks (classifier)
make bench

# Lint
make lint

# Format code
make fmt

# Build Docker image
make docker-build

# Run with Docker
make docker-run
```

### Requirements

- Go 1.23+
- Docker (optional, for containerized deployment)

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
