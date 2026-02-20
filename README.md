# AION -- Intelligent LLM Cost Router

**Same quality. 40-70% cheaper.**

AION analyzes the complexity of your LLM requests and routes them to the cheapest model that can handle it. Simple questions go to cheap models. Complex reasoning goes to capable ones. You save money without sacrificing quality.

> AION is NOT another OpenRouter/LiteLLM. Those are dumb pipes -- you pick a model, they forward. AION is different: you send a request without specifying a model, and AION figures out the right one.

## Supported Providers

| Provider | Type | Auth |
|---|---|---|
| **OpenAI** | OpenAI-compatible | Bearer token |
| **Anthropic** | Messages API (translated internally) | API key |
| **AWS Bedrock** | Anthropic Messages via Bedrock | Bearer token |
| **Google Vertex AI** | Anthropic Messages via Vertex | Bearer token |
| **Google Gemini** | OpenAI-compatible | Bearer token |
| **xAI Grok** | OpenAI-compatible | Bearer token |
| **OpenRouter** | OpenAI-compatible | Bearer token |

## Supported Ingress Formats

AION accepts requests in **both** OpenAI and Anthropic formats:

| Endpoint | Format | Use with |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | OpenAI SDK, LangChain, any OpenAI-compatible client |
| `POST /v1/messages` | Anthropic | Anthropic SDK, **Claude Code**, any Anthropic-compatible client |

Both endpoints go through the same routing pipeline: classify -> route -> budget check -> dispatch -> telemetry.

---

## Quick Start

### Option A: Docker Compose (recommended)

```bash
git clone https://github.com/ShubhamDX/aion.git
cd aion

# 1. Create your config
cp configs/aion.example.yaml configs/aion.yaml
# Edit configs/aion.yaml -- uncomment providers you want, add API keys

# 2. (Optional) Create .env for secrets
cat > .env <<EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
XAI_API_KEY=...
EOF

# 3. Run
docker compose up --build -d

# 4. Verify
curl http://localhost:8080/health
```

### Option B: From source

```bash
git clone https://github.com/ShubhamDX/aion.git
cd aion

# Build
go build -o aion ./cmd/aion

# Configure
cp configs/aion.example.yaml configs/aion.yaml
# Edit configs/aion.yaml

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run
./aion -config configs/aion.yaml
```

### Option C: Docker Hub

```bash
# Pull the image
docker pull shubhamdx/aion:latest    # or shubhamdx/aion:0.2.0

# Grab the example config
curl -O https://raw.githubusercontent.com/ShubhamDX/aion/main/configs/aion.example.yaml
cp aion.example.yaml configs/aion.yaml
# Edit configs/aion.yaml -- uncomment providers, add API keys

# Run
docker run -d \
  --name aion \
  -p 8080:8080 \
  -v $(pwd)/configs/aion.yaml:/app/configs/aion.yaml:ro \
  -v aion-data:/app/data \
  --env-file .env \
  shubhamdx/aion:latest

# Verify
curl http://localhost:8080/health
```

---

## Usage

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-aion-dev-key-change-me",  # your AION key
)

# Auto-routed -- AION picks the cheapest model that works
response = client.chat.completions.create(
    model="aion-auto",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(response.choices[0].message.content)
```

### OpenAI SDK (JavaScript)

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "sk-aion-dev-key-change-me",
});

const response = await client.chat.completions.create({
  model: "aion-auto",
  messages: [{ role: "user", content: "What is 2+2?" }],
});
console.log(response.choices[0].message.content);
```

### Anthropic SDK (Python)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="sk-aion-dev-key-change-me",  # your AION key
)

message = client.messages.create(
    model="aion-auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(message.content[0].text)
```

### Claude Code

Point Claude Code at AION and let it auto-route:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=sk-aion-dev-key-change-me
export ANTHROPIC_MODEL=aion-auto

# Make sure Bedrock mode is off
unset CLAUDE_CODE_USE_BEDROCK

claude
```

Simple messages (greetings, quick questions) route to cheap models like Haiku. Complex tasks (architecture, multi-file refactors) route to capable models like Opus. You pay for what you need.

### curl

```bash
# OpenAI format
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-aion-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"aion-auto","messages":[{"role":"user","content":"hello"}]}'

# Anthropic format
curl http://localhost:8080/v1/messages \
  -H "x-api-key: sk-aion-dev-key-change-me" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"aion-auto","max_tokens":256,"messages":[{"role":"user","content":"hello"}]}'

# Streaming (Anthropic format)
curl -N http://localhost:8080/v1/messages \
  -H "x-api-key: sk-aion-dev-key-change-me" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"aion-auto","max_tokens":256,"messages":[{"role":"user","content":"hello"}],"stream":true}'
```

---

## How It Works

```
              +-----------+        +-----------+
              | OpenAI    |        | Anthropic |
              | SDK/curl  |        | SDK/Claude|
              +-----+-----+        +-----+-----+
                    |                     |
          POST /v1/chat/         POST /v1/messages
          completions                     |
                    |                     |
                    +--------+--------+
                             |
                       +-----v------+
                       |    Auth    |  Bearer token or x-api-key
                       +-----+------+
                             |
                       +-----v------+
                       | Classifier |  <1ms overhead
                       | (7 signals)|
                       +-----+------+
                             |
                  +----------+----------+
                  |          |          |
             Tier 1     Tier 2     Tier 3
            (simple)  (moderate) (complex)
                  |          |          |
              +---v---+  +---v---+  +---v---+
              |Cheapest|  |Cheapest|  |Cheapest|
              |model   |  |model   |  |model   |
              +---+---+  +---+---+  +---+---+
                  |          |          |
        +---------+----------+---------+
        |         |          |         |
     OpenAI  Anthropic  Bedrock   Gemini  ...
        |         |          |         |
                  +-----+-----+
                        |
                  +-----v------+
                  | Telemetry  |  async, batched
                  |  (SQLite)  |
                  +------------+
```

### Classifier Signals

The classifier scores each request's complexity in under 1ms using 7 weighted signals:

| Signal | What it measures | Weight |
|---|---|---|
| **Content keywords** | Complexity verbs in last user message (analyze, implement, debug...) | 0.25 |
| **Intent (ML)** | TF-IDF + logistic regression on user message | 0.35 |
| **Token volume** | Content length (excluding system prompt) | 0.10 |
| **Message count** | Conversation turns | 0.05 |
| **System prompt** | Strong complexity keywords only (not length) | 0.05 |
| **Tool presence** | Binary: tools present or not | 0.05 |
| **User hints** | `aion_preferences` field in request | 0.15 |

The classifier produces a score between 0.0 and 1.0:

- **Score < 0.35** -- Tier 1 (simple): cheapest models (gpt-4o-mini, Haiku, Gemini Flash)
- **Score 0.35 - 0.70** -- Tier 2 (moderate): mid-range models (gpt-4o, Sonnet, Gemini Pro)
- **Score > 0.70** -- Tier 3 (complex): most capable models (o1, Opus, Grok)

The classifier is tuned for agentic clients like Claude Code -- it strips `<system-reminder>` tags and focuses on the actual user message, not boilerplate scaffolding.

### Virtual Models

| Model | Behavior |
|---|---|
| `aion-auto` | Classify and route to cheapest suitable model |
| `aion-escalate` | Force Tier 3 (most capable) |
| `<specific-model>` | Bypass classification, route directly |

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Supports streaming (`stream: true`).

**Auth:** `Authorization: Bearer <key>` or `x-api-key: <key>`

### `POST /v1/messages`

Anthropic-compatible Messages API. Supports streaming (`stream: true`).

**Auth:** `x-api-key: <key>` or `Authorization: Bearer <key>`

**Response headers (both endpoints):**

| Header | Description |
|---|---|
| `X-AION-Model` | The model that handled the request |
| `X-AION-Tier` | Complexity tier assigned (1, 2, or 3) |
| `X-AION-Cost-USD` | Estimated cost for this request |
| `X-AION-Savings-USD` | Estimated savings vs. most expensive model |
| `X-Request-ID` | Unique request identifier |

### `GET /v1/models`

Lists all available models across configured providers, plus AION virtual models.

### `GET /health`

Returns `200 OK` with version info. No auth required.

### `GET /aion/v1/metrics/savings`

Cost savings report over a time range.

### `GET /aion/v1/metrics/routing`

Routing distribution (how many requests went to each tier/model).

### `GET /aion/v1/metrics/costs`

Cost breakdown by provider and model.

---

## Configuration

Configuration lives in `configs/aion.yaml`. See [`configs/aion.example.yaml`](configs/aion.example.yaml) for a fully documented example.

### Providers

Configure one or more providers. Each provider lists models with tier assignments and per-token pricing:

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    models:
      - id: "gpt-4o-mini"
        tier: 1
        input_price_per_1m: 0.15
        output_price_per_1m: 0.60
      - id: "gpt-4o"
        tier: 2
        input_price_per_1m: 2.50
        output_price_per_1m: 10.00

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      - id: "claude-haiku-3-5"
        tier: 1
        input_price_per_1m: 0.80
        output_price_per_1m: 4.00

  gemini:
    api_key: "${GEMINI_API_KEY}"
    models:
      - id: "gemini-2.0-flash"
        tier: 1
        input_price_per_1m: 0.10
        output_price_per_1m: 0.40

  grok:
    api_key: "${XAI_API_KEY}"
    models:
      - id: "grok-3-mini"
        tier: 1
        input_price_per_1m: 0.30
        output_price_per_1m: 0.50

  bedrock:
    api_key: "${AWS_BEARER_TOKEN_BEDROCK}"
    region: "us-east-1"
    models:
      - id: "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        tier: 1
        input_price_per_1m: 0.80
        output_price_per_1m: 4.00
```

### Auth

```yaml
auth:
  enabled: true
  keys:
    - key: "sk-aion-dev-key-change-me"
      name: "development"
      budget:
        daily_limit_usd: 10.0
        monthly_limit_usd: 100.0
```

### Routing

```yaml
routing:
  strategy: "cheapest"        # cheapest or fallback
  classifier:
    tier1_threshold: 0.35     # score < 0.35 -> Tier 1
    tier2_threshold: 0.70     # score 0.35-0.70 -> Tier 2, > 0.70 -> Tier 3
  fallback_enabled: true      # fall back to adjacent tiers if no model available
```

### Environment Variables

API keys should be set as environment variables, not hardcoded:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."
export AWS_BEARER_TOKEN_BEDROCK="..."
```

The config file supports `${VAR}` and `$VAR` syntax for expansion.

---

## Docker

### docker-compose.yml

```bash
docker compose up --build -d     # build and start
docker compose logs -f           # follow logs
docker compose down              # stop
```

The compose file:
- Builds from the included Dockerfile
- Bind-mounts `configs/aion.yaml` (read-only)
- Persists SQLite telemetry data in a named volume
- Reads secrets from `.env`
- Health checks `/health` every 10s

### Logs

AION logs every routing decision:

```
INFO routed request_id=abc-123 ingress=anthropic requested_model=aion-auto
     routed_model=claude-haiku-4-5 provider=bedrock tier=1 score=0.078 stream=true
INFO request method=POST path=/v1/messages status=200 duration=1.2s
```

---

## Building from Source

```bash
# Build
go build -o aion ./cmd/aion

# Test
go test ./...

# Vet
go vet ./...
```

### Requirements

- Go 1.25+
- Docker (optional)

---

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
