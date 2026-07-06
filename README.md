<div align="center">

# AION

**Deterministic LLM routing. Pay the tier that fits the task, not the tier you forgot to downgrade.**

[![Go Version](https://img.shields.io/badge/go-1.25%2B-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-shubhamdx%2Faion-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/shubhamdx/aion)
[![Classifier](https://img.shields.io/badge/classify-%3C1ms-brightgreen)]()

A single Go binary that sits in front of your LLM providers, scores each request in **<1ms** and dispatches it to the lowest configured tier that matches the task. No GPU. No external calls. No code changes. Point your OpenAI or Anthropic SDK at `http://localhost:8080` and go.

</div>

---

## Why

Most agentic tools default to the strongest model for **every** turn. Real agent sessions contain a mix of task shapes: quick checks, file reads, small edits, debugging and larger design work. Routing all of them to the same premium model is structural overspend.

AION makes model selection proportional to actual task complexity.

```
"hello"                              ->  Tier 1  simple
"fix this typo"                      ->  Tier 1  simple
"add a null check on line 42"        ->  Tier 2  moderate
"refactor this package to use X"     ->  Tier 3  complex
```

AION records routing, cost and model decisions so teams can measure savings against their own traffic instead of relying on generic claims.

> AION is **not** OpenRouter or LiteLLM. Those forward requests to whichever model you specify. AION **decides** which model to use: you send `model: "aion-auto"`, the classifier picks the tier.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Local Inference](#local-inference): $0 Tier 1 via llama.cpp
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Docker](#docker)
- [Benchmarks](#benchmarks)
- [Open Core](#open-core) · [Roadmap](#roadmap) · [License](#license)

---

## Supported Providers

| Provider | Ingress format | Auth |
|---|---|---|
| **OpenAI** | OpenAI-compatible | Bearer token |
| **Anthropic** | Messages API (translated internally) | API key |
| **AWS Bedrock** | Anthropic Messages via Bedrock | Bearer token |
| **Google Vertex AI** | Anthropic Messages via Vertex | Bearer token |
| **Google Gemini** | OpenAI-compatible | Bearer token |
| **xAI Grok** | OpenAI-compatible | Bearer token |
| **OpenRouter** | OpenAI-compatible | Bearer token |
| **Local (llama.cpp)** | llama-server (OpenAI-compatible) | none · always **$0** |

### Ingress Endpoints

| Endpoint | Format | Use with |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | OpenAI SDK · LangChain · any OpenAI client |
| `POST /v1/messages` | Anthropic | Anthropic SDK · **Claude Code** · any Anthropic client |

Both pipelines converge on the same core: **classify -> route -> budget-check -> dispatch -> telemetry.**

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/ShubhamDX/aion.git && cd aion

# 1. Configure providers and API keys
cp configs/aion.example.yaml configs/aion.yaml
cp .env.example .env                       # add your provider keys

# 2. Run
docker compose up --build -d

# 3. Verify
curl http://localhost:8080/health
```

### Option B — From source

```bash
go build -o aion ./cmd/aion
cp configs/aion.example.yaml configs/aion.yaml
export OPENAI_API_KEY="sk-..." ANTHROPIC_API_KEY="sk-ant-..."
./aion -config configs/aion.yaml
```

### Option C — Docker Hub

```bash
docker pull shubhamdx/aion:latest         # or :0.3.0

docker run -d --name aion -p 8080:8080 \
  -v $(pwd)/configs/aion.yaml:/app/configs/aion.yaml:ro \
  -v aion-data:/app/data \
  --env-file .env \
  shubhamdx/aion:latest
```

---

## Usage

<details open>
<summary><b>Python · OpenAI SDK</b></summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-aion-dev-key-change-me",
)

resp = client.chat.completions.create(
    model="aion-auto",                     # let AION pick the tier
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(resp.choices[0].message.content)
```
</details>

<details>
<summary><b>JavaScript · OpenAI SDK</b></summary>

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "sk-aion-dev-key-change-me",
});

const resp = await client.chat.completions.create({
  model: "aion-auto",
  messages: [{ role: "user", content: "What is 2+2?" }],
});
```
</details>

<details>
<summary><b>Python · Anthropic SDK</b></summary>

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="sk-aion-dev-key-change-me",
)

msg = client.messages.create(
    model="aion-auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
```
</details>

<details>
<summary><b>Claude Code</b></summary>

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=sk-aion-dev-key-change-me
export ANTHROPIC_MODEL=aion-auto
unset CLAUDE_CODE_USE_BEDROCK

claude
```

Trivial messages (greetings, quick questions) route to Haiku. Multi-file refactors route to Opus. You pay for what you need.
</details>

<details>
<summary><b>curl</b></summary>

```bash
# OpenAI
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-aion-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"aion-auto","messages":[{"role":"user","content":"hello"}]}'

# Anthropic + streaming
curl -N http://localhost:8080/v1/messages \
  -H "x-api-key: sk-aion-dev-key-change-me" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"aion-auto","max_tokens":256,"stream":true,"messages":[{"role":"user","content":"hello"}]}'
```
</details>

---

## How It Works

```
      ┌────────────┐                ┌────────────┐
      │  OpenAI    │                │ Anthropic  │
      │  SDK/curl  │                │ SDK/Claude │
      └─────┬──────┘                └─────┬──────┘
            │                             │
   POST /v1/chat/completions     POST /v1/messages
            │                             │
            └──────────────┬──────────────┘
                           ▼
                   ┌──────────────┐
                   │     Auth     │   Bearer / x-api-key
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  Classifier  │   7 signals · <1ms
                   └──────┬───────┘
                          ▼
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Tier 1         Tier 2         Tier 3
       simple         moderate       complex
          │              │              │
          ▼              ▼              ▼
       ┌──────┐       ┌──────┐       ┌──────┐
       │Local │       │ gpt  │       │ Opus │
       │ Qwen │       │ -4o  │       │  o1  │
       │Haiku │       │Sonnet│       │ Grok │
       │ ...  │       │ ...  │       │ ...  │
       └───┬──┘       └───┬──┘       └───┬──┘
           └──────────────┼──────────────┘
                          ▼
                   ┌──────────────┐
                   │   Telemetry  │   async · SQLite · local-only
                   └──────────────┘
```

### Classifier Signals

Each request gets a complexity score in [0, 1] from 7 weighted signals:

| Signal | What it measures | Weight |
|---|---|---|
| **Content keywords** | Complexity verbs in last user message (`analyze`, `implement`, `debug`…) | 0.25 |
| **Intent (ML)** | TF-IDF + logistic regression on the user message | 0.35 |
| **Token volume** | Content length, excluding system prompt | 0.10 |
| **Message count** | Conversation turn depth | 0.05 |
| **System prompt** | Strong complexity keywords only (not length) | 0.05 |
| **Tool presence** | Binary — tools attached or not | 0.05 |
| **User hints** | `aion_preferences` field in request | 0.15 |

```
score < 0.35         ─►  Tier 1  ─►  cheap    (gpt-4o-mini, Haiku, Flash, Local)
0.35 ≤ score ≤ 0.70  ─►  Tier 2  ─►  mid      (gpt-4o, Sonnet, Pro)
score > 0.70         ─►  Tier 3  ─►  capable  (o1, Opus, Grok)
```

The classifier is tuned for agentic clients — it strips `<system-reminder>` scaffolding and focuses on the actual user turn.

### Confirmation-Aware Escalation

Short confirmations ("yes", "do it", "go ahead") would normally score as Tier 1. But in context, they're often green-lighting a complex plan the assistant just proposed.

```
User: "refactor the entire auth system with JWT refresh token rotation"
Assistant: [proposes 3-step plan with code blocks]
User: "do it"

Without escalation:  score=0.12 → Tier 1 (Haiku)    ✗ wrong model
With escalation:     score=0.85 → Tier 3 (Opus)    ✓ correct
```

~35 confirmation patterns are recognized. Escalation only fires when the preceding assistant turn shows complexity signals (code blocks, multi-step plans, long responses). A "yes" after "Hi, how can I help?" stays Tier 1.

### Virtual Models

| Model | Behavior |
|---|---|
| `aion-auto` | Classify and route to the cheapest healthy model |
| `aion-local` | Force local llama.cpp (Tier 1, $0) |
| `aion-escalate` | Force Tier 3 |
| `<specific-model-id>` | Bypass classification, route directly |

---

## Local Inference

AION can serve Tier 1 at **$0** by routing to a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server. Great for privacy-sensitive workloads, air-gapped deployments, or squeezing the last cent out of your bill.

### Sidecar mode (recommended)

Ships as a companion container. First run auto-downloads a GGUF into a named volume — subsequent restarts reuse it.

```bash
# Pulls llama-server + downloads Qwen2.5-1.5B-Instruct (~1GB) on first run
docker compose --profile local up -d
```

Enable in `configs/aion.yaml`:

```yaml
providers:
  local:
    enabled: true
    base_url: "http://llama-server:8081/v1"
    models:
      - id: "qwen2.5-1.5b-instruct"
        tier: 1
```

Override the model with env vars in `.env`:

```bash
LLAMA_MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF
LLAMA_MODEL_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### Managed mode (dev / single-node)

AION spawns `llama-server` as a subprocess. Useful when `llama-server` is installed locally.

```yaml
providers:
  local:
    enabled: true
    models:
      - id: "qwen2.5-1.5b-instruct"
        tier: 1
    managed:
      binary_path: "llama-server"
      model_path: "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
      port: 8081
      threads: 4
      ctx_size: 4096
      ready_timeout: "120s"
```

Pricing is **force-zeroed** in both modes — `aion-auto` picks local first for any Tier 1 request. Force it explicitly with `model: "aion-local"`.

---

## API Reference

### Endpoints

| Route | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions (streaming supported) |
| `POST /v1/messages` | Anthropic-compatible Messages API (streaming supported) |
| `GET  /v1/models` | All models across configured providers, plus AION virtual models |
| `GET  /health` | Liveness probe — returns `200 OK` with version, no auth |
| `GET  /aion/v1/metrics/savings` | Cost savings over a time range |
| `GET  /aion/v1/metrics/routing` | Request distribution across tiers and models |
| `GET  /aion/v1/metrics/costs` | Cost breakdown by provider and model |

### Response Headers (all ingresses)

| Header | Description |
|---|---|
| `X-AION-Model` | The model that handled the request |
| `X-AION-Provider` | Provider the request was dispatched to (`openai`, `anthropic`, `local`, …) |
| `X-AION-Tier` | Complexity tier assigned (`1`, `2`, `3`) |
| `X-AION-Cost-USD` | Estimated cost for this request |
| `X-AION-Savings-USD` | Estimated savings vs. the most expensive configured model |
| `X-Request-ID` | Unique request identifier |

### Auth

`Authorization: Bearer <key>` or `x-api-key: <key>` — either works on either ingress.

---

## Configuration

Config lives in `configs/aion.yaml`. See [`configs/aion.example.yaml`](configs/aion.example.yaml) for an annotated template.

### Providers

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    models:
      - { id: "gpt-4o-mini", tier: 1, input_price_per_1m: 0.15, output_price_per_1m: 0.60 }
      - { id: "gpt-4o",      tier: 2, input_price_per_1m: 2.50, output_price_per_1m: 10.00 }
      - { id: "o1",          tier: 3, input_price_per_1m: 15.00, output_price_per_1m: 60.00 }

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      - { id: "claude-haiku-3-5", tier: 1, input_price_per_1m: 0.80, output_price_per_1m: 4.00 }
      - { id: "claude-sonnet-4",  tier: 2, input_price_per_1m: 3.00, output_price_per_1m: 15.00 }
      - { id: "claude-opus-4",    tier: 3, input_price_per_1m: 15.00, output_price_per_1m: 75.00 }
```

Environment variables: `${VAR}` and `$VAR` expand at load time — API keys never live in the file.

### Auth + Budgets

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

Per-key daily/monthly caps enforced on every request. `429 rate_limit_error` returned when exceeded.

### Routing

```yaml
routing:
  strategy: "cheapest"          # cheapest | fallback
  classifier:
    tier1_threshold: 0.35       # score < 0.35  → Tier 1
    tier2_threshold: 0.70       # 0.35 ≤ score ≤ 0.70 → Tier 2, > 0.70 → Tier 3
  fallback_enabled: true        # escalate to a higher tier if no healthy model in current tier
```

### Telemetry

```yaml
telemetry:
  db_path: "./data/aion.db"
  batch_size: 100
  flush_interval: "5s"
```

Telemetry is async, batched, and stays on-disk. Nothing leaves your infrastructure.

---

## Docker

```bash
docker compose up --build -d          # build + start
docker compose --profile local up -d  # + local llama.cpp sidecar
docker compose logs -f
docker compose down
```

The compose file bind-mounts `configs/aion.yaml` read-only, persists SQLite telemetry in a named volume, reads secrets from `.env`, and health-checks `/health` every 10s.

### Logs

```
INFO routed request_id=abc-123 ingress=anthropic requested_model=aion-auto
     routed_model=claude-haiku-4-5 provider=bedrock tier=1 score=0.078 stream=true
INFO request method=POST path=/v1/messages status=200 duration=1.2s
```

---

## Building from Source

```bash
go build -o aion ./cmd/aion
go test ./...
go vet ./...
```

**Requirements:** Go 1.25+, Docker (optional).

---

## Benchmarks

Classifier benchmarks — a 1000-prompt workload mix and a 200-step autonomous session simulation — live in `internal/classifier/benchmark_test.go`:

```bash
go test ./internal/classifier/... -run TestBenchmark -v
```

Benchmarks validate routing behavior, not output quality. Actual savings depend on your workload distribution.

---

## Open Core

AION's routing engine and classifier are fully open source (Apache 2.0).

Adaptive learning, hosted analytics, and enterprise governance may ship as optional external services. **The core routing logic will always remain open.**

## Roadmap

- [ ] Adaptive routing based on regeneration signals (auto-escalate on retry)
- [ ] Latency-aware routing (factor provider response times, not just price)
- [ ] Token-budget targeting ("best answer for $0.01")
- [ ] Routing analytics dashboard
- [ ] Cross-session learning from misroutes
- [ ] Enterprise governance and policy controls

## License

[Apache 2.0](LICENSE)

<div align="center">
<sub>Built with ❤️ for teams tired of paying Opus prices for "fix this typo."</sub>
</div>
