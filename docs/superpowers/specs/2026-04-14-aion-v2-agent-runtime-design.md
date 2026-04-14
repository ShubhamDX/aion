# AION v2 — Universal AI Gateway & Agent Runtime

**Date:** 2026-04-14
**Status:** Approved
**Scope:** FastText classifier, exact-match cache, DAG chain executor, ReAct agent runtime, AION-native OpenAPI

---

## 1. System Overview

AION v2 is a universal AI gateway and agent runtime that:

1. Accepts requests through a single AION-native API (OpenAPI documented)
2. Classifies complexity using a FastText ML model (<1ms CPU)
3. Routes each request/step to the optimal model across any provider
4. Supports three execution modes: single call, DAG chain, ReAct agent
5. Caches exact-match queries to eliminate redundant LLM calls

### Architecture

```
Client SDK (Python/JS/Go — auto-generated from OpenAPI spec)
                              |
                    +-------- v ----------+
                    |   AION Gateway      |
                    |  +---------------+  |
                    |  | Auth + Budget  |  |
                    |  +-------+-------+  |
                    |          v          |
                    |  +---------------+  |
                    |  | Exact Cache   |---> HIT: return cached response ($0)
                    |  +-------+-------+  |
                    |          v MISS     |
                    |  +---------------+  |
                    |  |  Execution    |  |
                    |  |  Engine       |  |
                    |  |               |  |
                    |  | single|chain  |  |
                    |  |    agent      |  |
                    |  +-------+-------+  |
                    |          v          |
                    |  +---------------+  |
                    |  | Core Pipeline |  |  <-- shared by all 3 modes
                    |  |               |  |
                    |  | classify      |  |  <-- FastText (<1ms)
                    |  | route         |  |  <-- cheapest healthy model
                    |  | translate     |  |  <-- provider-specific format
                    |  | dispatch      |  |  <-- call provider
                    |  | record        |  |  <-- telemetry + feedback
                    |  +-------+-------+  |
                    +----------+-----------+
                               v
               +-------+-------+-------+--------+
               |Anthropic|OpenAI|Google |Local   | ... any provider
               +-------+-------+-------+--------+
```

---

## 2. API Design

AION-native API with OpenAPI spec. Not OpenAI-compatible — provider translation is purely internal.

### Endpoints

```
POST /v1/run          -> single call (auto-routed)
POST /v1/chain        -> DAG execution
POST /v1/agent        -> ReAct loop
GET  /v1/models       -> list available models across all providers
GET  /v1/usage        -> cost/savings telemetry
POST /v1/retrain      -> retrain classifier from updated dataset (admin, deferred to enterprise)
GET  /v1/health       -> health check
GET  /docs            -> Swagger UI (auto-generated)
```

### Single Call — POST /v1/run

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
  ],
  "preferences": {
    "tier": "auto",
    "max_cost_usd": 0.05,
    "max_latency_ms": 2000
  },
  "stream": false,
  "tools": []
}
```

`preferences.tier` options: `"auto"` (default), `"tier1"`, `"tier2"`, `"tier3"`.
Optional overrides: `preferences.model` and `preferences.provider` to force a specific model.

### DAG Chain — POST /v1/chain

```json
{
  "steps": [
    {
      "id": "extract",
      "messages": [
        {"role": "user", "content": "Extract key entities from: {{input}}"}
      ],
      "preferences": {"tier": "auto"}
    },
    {
      "id": "analyze",
      "depends_on": ["extract"],
      "messages": [
        {"role": "user", "content": "Analyze trends in: {{extract.output}}"}
      ]
    },
    {
      "id": "summarize",
      "depends_on": ["extract"],
      "messages": [
        {"role": "user", "content": "Summarize: {{extract.output}}"}
      ]
    },
    {
      "id": "report",
      "depends_on": ["analyze", "summarize"],
      "messages": [
        {"role": "user", "content": "Combine into report: {{analyze.output}} and {{summarize.output}}"}
      ]
    }
  ],
  "input": "... raw document text ...",
  "preferences": {
    "max_total_cost_usd": 0.50
  },
  "on_step_failure": "abort"
}
```

Steps with shared dependencies run in parallel. Template variables `{{step_id.output}}` are resolved from completed step outputs. `on_step_failure` options: `"abort"` (default), `"skip"`, `"retry"` (retries one tier higher).

### ReAct Agent — POST /v1/agent

```json
{
  "goal": "Find Q4 revenue and compare with last year",
  "tools": [
    {
      "name": "db_query",
      "description": "Run a SQL query against the financials database",
      "execution": "server",
      "endpoint": "https://internal.company.com/tools/db_query",
      "parameters": {
        "type": "object",
        "properties": {
          "sql": {"type": "string"}
        }
      }
    },
    {
      "name": "web_search",
      "description": "Search the web",
      "execution": "client",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        }
      }
    }
  ],
  "max_iterations": 10,
  "preferences": {
    "max_total_cost_usd": 1.00
  }
}
```

Tool execution modes:
- `"server"`: AION calls the tool endpoint directly. Requires `endpoint` field.
- `"client"`: AION streams a `tool_call` SSE event. Client executes the tool and posts result to `POST /v1/agent/{run_id}/tool_result`.

### Unified Response Format

```json
{
  "id": "aion_req_abc123",
  "mode": "single",
  "status": "completed",
  "output": "Python is a high-level programming language...",
  "steps": [
    {
      "id": "step_0",
      "tier": 1,
      "model": "gpt-4o-mini",
      "provider": "openai",
      "input_tokens": 24,
      "output_tokens": 85,
      "cost_usd": 0.00006,
      "latency_ms": 340,
      "cached": false
    }
  ],
  "total_cost_usd": 0.00006,
  "savings_usd": 0.00142,
  "total_latency_ms": 340
}
```

For chains/agents, the `steps` array contains every step with its routing decision.

### Streaming (SSE)

For chains and agents, events stream per-step:

```
event: step_start
data: {"step_id": "extract", "tier": 1, "model": "gpt-4o-mini"}

event: step_delta
data: {"step_id": "extract", "delta": "Key entities found: ..."}

event: step_complete
data: {"step_id": "extract", "cost_usd": 0.0001}

event: step_start
data: {"step_id": "analyze", "tier": 3, "model": "claude-sonnet-4-20250514"}
...

event: done
data: {"total_cost_usd": 0.0043, "savings_usd": 0.0127}
```

For ReAct agent with client-side tools:

```
event: tool_call
data: {"tool": "db_query", "input": {"sql": "..."}, "call_id": "abc"}
```

Client responds: `POST /v1/agent/{run_id}/tool_result {"call_id": "abc", "result": "..."}`

---

## 3. FastText Classifier

### Purpose

Replace the heuristic classifier (keyword matching + weighted signals) with a trained FastText model that learns routing patterns from labeled data. <1ms inference on CPU.

### How it works

1. Input: last user message + system prompt (concatenated)
2. FastText embeds words + character n-grams, averages vectors
3. Linear classifier outputs: `{tier1: 0.08, tier2: 0.84, tier3: 0.08}`
4. Route to highest-probability tier if confidence >= threshold
5. Fall back to heuristic classifier if confidence < threshold

### Training data

Source: 500K queries from public datasets (ShareGPT, LMSYS Chatbot Arena, WildChat, OpenAssistant).

Labeling method: LLM-as-judge via Bedrock. For each query, determine the minimum tier that produces an acceptable response:
1. Send to Tier 1 model. Judge quality.
2. If quality acceptable -> label as tier1. Done.
3. If not -> send to Tier 2. Judge quality.
4. If acceptable -> label as tier2. Done.
5. If not -> label as tier3.

Cost: one-time ~$500-1000 for labeling 500K queries via Bedrock.

Result: ~200K tier1, ~200K tier2, ~100K tier3 labeled examples.

### Training format

```
__label__tier1 What is 2+2?
__label__tier2 Translate this paragraph to French
__label__tier3 Analyze this codebase and propose a refactoring strategy
__label__tier1 [step:planning] Break down this task into subtasks
__label__tier3 [step:research] Analyze competitive landscape for fintech in SEA
__label__tier1 [step:react_think] I need to figure out what tool to call next
```

The `[step:type]` prefix is a feature for chain/agent context-aware classification.

Training time: <60 seconds for 500K examples. Model file: ~15-20MB.

### Cold start and fallback

1. Ship a pre-trained default model (`classifier.bin`) inside the Docker image
2. If FastText confidence < 0.6, fall back to existing heuristic classifier
3. No runtime learning required — classifier works from day one

### Integration

```go
type Classifier interface {
    Classify(messages []Message, stepCtx *StepContext) ClassifyResult
}

type FastTextClassifier struct {
    model    *fasttext.Model
    fallback Classifier        // heuristic
    minConf  float64           // 0.6 default
}

func (f *FastTextClassifier) Classify(messages []Message, stepCtx *StepContext) ClassifyResult {
    text := extractText(messages)
    if stepCtx != nil {
        text = fmt.Sprintf("[step:%s] %s", stepCtx.StepType, text)
    }

    prediction := f.model.Predict(text)
    if prediction.Confidence < f.minConf {
        return f.fallback.Classify(messages, stepCtx)
    }

    return ClassifyResult{
        Tier:       prediction.Tier,
        Confidence: prediction.Confidence,
        Method:     "fasttext",
    }
}
```

### Go bindings

Use `go-fasttext` (CGo wrapper around Facebook's C++ library). Model loads once at startup (~50ms). Inference is lock-free and thread-safe.

### Per-customer fine-tuning

Deferred to enterprise version. The default model handles 85-90% of query patterns.

---

## 4. Exact-Match Cache

### Purpose

Cache full LLM responses for identical queries. No embeddings, no ML — pure hash-based lookup.

### How it works

1. Compute cache key: `SHA-256(system_prompt + last_user_message + tool_names + tier)`
2. Lookup in hash map: O(1)
3. HIT: return cached response. Cost: $0. Latency: <1ms.
4. MISS: proceed through core pipeline. Cache the response after.

### What gets cached

Only single-call, non-streaming, auto-routed, single-turn requests. This ensures high cache quality and avoids caching context-dependent responses.

NOT cached:
- Multi-turn conversations (>1 user message) — context-dependent
- Streaming requests — cache stores complete responses only
- ReAct agent iterations — each iteration is unique
- Requests with forced model — cache is tier-based

### Eviction

- **TTL**: Entries expire after configurable duration (default: 1 hour)
- **LRU**: When full, evict least recently used entry
- **Max size**: Configurable cap (default: 50K entries, ~250-400MB RAM)
- **Per API key isolation**: Each key gets its own cache namespace

### Data structure

```go
type Cache struct {
    entries  map[string]*CacheEntry  // hash -> entry
    lru      *list.List              // eviction order
    maxSize  int
    ttl      time.Duration
    mu       sync.RWMutex
}

type CacheEntry struct {
    Key       string
    Response  *StepResult
    CreatedAt time.Time
    lruElem   *list.Element
}
```

### Metrics

Exposed via telemetry:
- cache_hits, cache_misses, cache_hit_rate
- cache_savings_usd, cache_entries (current count)

---

## 5. DAG Chain Executor

### Purpose

Execute a directed acyclic graph of LLM steps with per-step routing and parallel execution.

### Execution flow

1. Validate DAG (no cycles, all dependencies exist, all templates reference valid step IDs)
2. Find steps with no dependencies -> launch in parallel
3. As each step completes, check if any dependent steps are now unblocked
4. Launch newly unblocked steps in parallel
5. Continue until all steps complete, budget is exhausted, or an error occurs

### Step execution

Each step calls `CorePipeline.RunStep()`:
1. Resolve template variables (`{{step_id.output}}` -> actual output)
2. Check exact-match cache
3. Classify + Route + Dispatch via core pipeline
4. Record telemetry with chain context

### Parallel execution

Independent steps (shared dependency only) run concurrently via goroutines. Max concurrency is configurable (`chain.max_parallel`, default: 10).

### Budget enforcement

Total chain budget (`max_total_cost_usd`). Before each step, check remaining budget. If estimated cost exceeds remaining budget, abort and return partial results.

### Error handling

Three modes, configurable per chain via `on_step_failure`:
- **abort** (default): Stop entire chain. Return completed steps.
- **skip**: Skip failed step. Dependent steps receive empty input for that dependency.
- **retry**: Retry failed step at one tier higher. If Tier 1 failed, retry at Tier 2.

---

## 6. ReAct Agent Executor

### Purpose

Run an autonomous think-act-observe loop. AION routes each iteration to the optimal tier independently.

### Loop structure

```
For each iteration (up to max_iterations):
  1. Send conversation history to LLM (via core pipeline, auto-routed)
  2. Parse response:
     - THINK + ACTION -> execute tool, add result to history, continue loop
     - THINK + ANSWER -> return final answer, exit loop
  3. Check budget and iteration limits
```

### ReAct system prompt

Injected by AION. Teaches the model the THINK/ACTION/ANSWER format, lists available tools with descriptions and parameter schemas.

### Tool execution

Two modes per tool:
- **Server-side** (`"execution": "server"`): AION POSTs to the tool's endpoint URL directly. Requires `endpoint` field. Timeout: configurable (default 30s).
- **Client-side** (`"execution": "client"`): AION streams a `tool_call` SSE event. Client executes locally and responds via `POST /v1/agent/{run_id}/tool_result`.

### Per-iteration routing

Each iteration is classified independently. Planning thoughts route to Tier 1 (cheap). Complex reasoning routes to Tier 2/3. This is the core cost optimization for agentic workloads.

### Conversation history management

As iterations accumulate, history grows. Sliding window strategy:
- Keep system prompt + last N iterations in full context (default N=8)
- Summarize older iterations into a single "progress so far" message
- Summarization step itself is routed via core pipeline (Tier 1 task)

### Safety guardrails

- `max_iterations`: Hard cap on loop count (default: 20)
- `max_cost_usd`: Hard cap on total cost (default: $5.00)
- `max_tool_calls`: Prevent infinite tool loops (default: 50)
- `tool_timeout`: Per-tool execution timeout (default: 30s)

If any limit is hit, return partial response with `stopped_reason` field.

---

## 7. Core Pipeline

The shared function called by all three execution modes.

```
Single call  --> calls RunStep once
DAG chain    --> calls RunStep per step
ReAct agent  --> calls RunStep per iteration
                      |
                      v
              +----------------+
              |    RunStep     |
              |                |
              | 1. Cache check | -> exact-match
              | 2. Classify    | -> FastText (<1ms)
              | 3. Route       | -> cheapest healthy model for tier
              | 4. Translate   | -> convert to provider format
              | 5. Dispatch    | -> call provider
              | 6. Record      | -> telemetry (async)
              | 7. Cache set   | -> store response
              +----------------+
```

### StepContext

Passed by chain/agent executors to inform classification:

```go
type StepContext struct {
    Mode       string // "single", "chain", "agent"
    StepID     string // e.g., "extract", "iteration_3"
    StepType   string // e.g., "planning", "research", "formatting"
    ChainID    string // groups steps from same chain/agent run
}
```

---

## 8. Package Structure

```
internal/
  core/
    pipeline.go          # CorePipeline with RunStep
  cache/
    cache.go             # Exact-match cache (LRU + TTL)
  classifier/
    classifier.go        # Classifier interface (unchanged)
    fasttext.go          # FastText implementation
    heuristic.go         # Existing heuristic (renamed, kept as fallback)
    signals.go           # Existing signals (used by heuristic)
  chain/
    executor.go          # DAG chain executor
    dag.go               # DAG validation + topological sort
    template.go          # {{step.output}} resolution
  agent/
    executor.go          # ReAct loop executor
    prompt.go            # ReAct system prompt builder
    parser.go            # Parse THINK/ACTION/ANSWER from response
    history.go           # Sliding window + summarization
  proxy/
    handler.go           # Simplified — delegates to core pipeline
    stream.go            # SSE streaming
  router/                # Unchanged
  provider/              # Unchanged
  pricing/               # Unchanged
  budget/                # Extended for per-chain budgets
  telemetry/             # Extended for step-level recording
  server/
    routes.go            # New routes: /v1/run, /v1/chain, /v1/agent
    openapi.go           # OpenAPI spec generation
```

---

## 9. Configuration

```yaml
server:
  port: 8080
  read_timeout: 30s
  write_timeout: 120s
  shutdown_timeout: 15s

auth:
  enabled: true
  keys:
    - key: "sk-aion-xxx"
      name: "dev-team"
      budget:
        daily_limit_usd: 50.0
        monthly_limit_usd: 500.0

providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      - id: claude-sonnet-4-20250514
        tier: 2
        input_price_per_1m: 3.0
        output_price_per_1m: 15.0
      - id: claude-haiku-3-5
        tier: 1
        input_price_per_1m: 0.25
        output_price_per_1m: 1.25
  openai:
    api_key: "${OPENAI_API_KEY}"
    models:
      - id: gpt-4o
        tier: 2
        input_price_per_1m: 2.5
        output_price_per_1m: 10.0
      - id: gpt-4o-mini
        tier: 1
        input_price_per_1m: 0.15
        output_price_per_1m: 0.60
  bedrock:
    region: "us-east-1"
    auth_token: "${BEDROCK_AUTH_TOKEN}"
    models:
      - id: anthropic.claude-sonnet-4-20250514-v1:0
        tier: 2
        input_price_per_1m: 3.0
        output_price_per_1m: 15.0
  local:
    enabled: false
    base_url: "http://localhost:8001"
    models:
      - id: llama-3.1-8b
        tier: 1
        input_price_per_1m: 0
        output_price_per_1m: 0

routing:
  strategy: "cheapest"
  classifier:
    type: "fasttext"
    model_path: "./models/classifier.bin"
    min_confidence: 0.6
    fallback: "heuristic"
  thresholds:
    tier1: 0.3
    tier2: 0.7

cache:
  enabled: true
  max_entries: 50000
  ttl: "1h"
  per_key_isolation: true

chain:
  max_steps: 50
  max_parallel: 10
  default_on_failure: "abort"

agent:
  max_iterations: 20
  max_cost_usd: 5.0
  max_tool_calls: 50
  tool_timeout: 30s
  history_window: 8
  summarize_older: true

telemetry:
  db_path: "./data/aion.db"
  batch_size: 100
  flush_interval: "5s"
```

---

## 10. Deployment

### Docker image contents

| Artifact | Size |
|---|---|
| `aion` binary | ~15MB |
| `classifier.bin` (FastText model) | ~15-20MB |
| `aion.yaml` (example config) | <1KB |
| **Total image** | **~50MB** |

### OpenAPI / SDK generation

Swagger UI auto-served at `GET /docs`. OpenAPI JSON at `GET /docs/openapi.json`. Client SDKs (Python, JS, Go) auto-generated from spec using `openapi-generator`.

---

## 11. Decisions & Deferred Items

### Decided

| Decision | Choice | Rationale |
|---|---|---|
| Classifier type | FastText | <1ms CPU, trains in seconds, 85-90% accuracy |
| Training data | 500K public dataset + Bedrock LLM-as-judge | One-time cost, no runtime evaluation needed |
| Cache type | Exact-match (SHA-256) | Simple, zero false positives, bounded |
| Agent runtime | DAG + ReAct | DAG for structured pipelines, ReAct for open-ended tasks |
| API style | AION-native + OpenAPI | Not locked to any provider's format |
| Tool execution | Server-side + client-side | Flexible for different deployment scenarios |

### Deferred to enterprise version

- Per-customer classifier fine-tuning
- Semantic cache (LSH-based, for paraphrased queries)
- Shadow evaluation for classifier improvement
- Runtime LLM-as-judge feedback loop
- Custom model training pipeline
