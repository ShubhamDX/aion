package classifier

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
)

// --- Prompt category definitions ---

type benchmarkCase struct {
	category string
	prompt   string
	messages []types.Message // full conversation (overrides prompt if set)
	tools    []types.Tool
}

// rawB wraps a string as a JSON-encoded json.RawMessage.
func rawB(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// --- Template pools for each category ---

func greetingPrompts() []string {
	return []string{
		"hello",
		"hi there",
		"hey",
		"good morning",
		"good afternoon",
		"good evening",
		"hi",
		"thanks",
		"thank you",
		"thanks a lot",
		"bye",
		"goodbye",
		"see you later",
		"how are you",
		"how's it going",
		"what's up",
		"yo",
		"howdy",
		"greetings",
		"hey there, how are you doing today",
		"thanks for the help",
		"cheers",
		"nice to meet you",
		"have a great day",
		"take care",
	}
}

func factualPrompts(rng *rand.Rand) []string {
	subjects := []string{
		"Python", "JavaScript", "Go", "Rust", "TypeScript", "Java", "C++",
		"Kubernetes", "Docker", "React", "PostgreSQL", "Redis", "gRPC",
		"TCP/IP", "HTTP/2", "WebSockets", "GraphQL", "REST",
	}
	people := []string{
		"Alan Turing", "Dennis Ritchie", "Linus Torvalds", "Grace Hopper",
		"Vint Cerf", "Tim Berners-Lee", "Ken Thompson", "Guido van Rossum",
	}
	templates := []string{
		"what is %s",
		"who created %s",
		"when was %s released",
		"define %s",
		"what does %s stand for",
		"what is the latest version of %s",
		"who invented %s",
		"list the main features of %s",
		"what language is %s written in",
		"what is %s used for",
		"who is %s",
		"when did %s start their career",
		"what is %s known for",
		"what are the basics of %s",
		"what is the difference between %s and %s",
	}

	var prompts []string
	for _, tmpl := range templates {
		count := strings.Count(tmpl, "%s")
		switch count {
		case 1:
			subj := subjects[rng.Intn(len(subjects))]
			prompts = append(prompts, fmt.Sprintf(tmpl, subj))
		case 2:
			s1 := subjects[rng.Intn(len(subjects))]
			s2 := subjects[rng.Intn(len(subjects))]
			prompts = append(prompts, fmt.Sprintf(tmpl, s1, s2))
		}
	}
	// Add people-based prompts
	for _, p := range people {
		prompts = append(prompts, fmt.Sprintf("who is %s", p))
		prompts = append(prompts, fmt.Sprintf("what did %s contribute to computer science", p))
	}
	return prompts
}

func simpleCodePrompts(rng *rand.Rand) []string {
	languages := []string{"Go", "Python", "JavaScript", "TypeScript", "Rust", "Java", "C++"}
	variables := []string{"count", "total", "result", "data", "items", "output", "value", "idx"}
	files := []string{"main.go", "app.py", "index.ts", "server.js", "lib.rs", "App.java"}

	templates := []string{
		"fix the typo on line 42 of %s",
		"rename variable %s to something more descriptive",
		"add a comment to the %s function",
		"remove the unused import in %s",
		"change the variable name from %s to userCount",
		"indent the code properly in %s",
		"add a newline at the end of %s",
		"fix the missing semicolon in %s",
		"update the copyright year in %s",
		"remove trailing whitespace in %s",
		"change the string literal to use %s template syntax",
		"capitalize the constant name %s",
		"make %s private instead of public",
		"add the return type annotation in %s",
		"fix the off-by-one error in the loop",
		"swap the order of these two lines",
		"add a blank line between functions",
		"convert tabs to spaces in %s",
		"remove the debug print statement",
		"fix the spelling in the error message",
	}

	var prompts []string
	for _, tmpl := range templates {
		if strings.Contains(tmpl, "%s") {
			pool := files
			if strings.Contains(tmpl, "variable") || strings.Contains(tmpl, "constant") {
				pool = variables
			} else if strings.Contains(tmpl, "syntax") || strings.Contains(tmpl, "private") {
				pool = languages
			}
			prompts = append(prompts, fmt.Sprintf(tmpl, pool[rng.Intn(len(pool))]))
		} else {
			prompts = append(prompts, tmpl)
		}
	}
	return prompts
}

func explanationPrompts(rng *rand.Rand) []string {
	concepts := []string{
		"closures", "goroutines", "channels", "mutex locks", "semaphores",
		"hash maps", "B-trees", "linked lists", "binary search", "quicksort",
		"TCP handshake", "DNS resolution", "TLS certificates", "JWT tokens",
		"dependency injection", "event loops", "garbage collection", "virtual memory",
		"load balancing", "sharding",
	}
	templates := []string{
		"explain how %s work",
		"what does %s do",
		"how does %s work in practice",
		"can you explain %s in simple terms",
		"what are %s and why are they useful",
		"explain %s like I'm a junior developer",
		"what is the purpose of %s",
		"how are %s implemented under the hood",
		"why do we need %s",
		"what happens when you use %s",
		"describe what %s means",
		"walk me through how %s works",
		"what's the concept behind %s",
		"tell me about %s",
		"how does %s relate to concurrency",
		"what role does %s play in web development",
	}

	var prompts []string
	for _, tmpl := range templates {
		c := concepts[rng.Intn(len(concepts))]
		prompts = append(prompts, fmt.Sprintf(tmpl, c))
	}
	return prompts
}

func codeGenPrompts(rng *rand.Rand) []string {
	languages := []string{"Go", "Python", "JavaScript", "TypeScript", "Rust", "Java"}
	structures := []string{
		"linked list", "binary tree", "hash map", "stack", "queue",
		"priority queue", "graph", "trie", "bloom filter", "ring buffer",
	}
	tasks := []string{
		"HTTP server", "CLI tool", "REST API endpoint", "database connection pool",
		"rate limiter", "retry mechanism", "worker pool", "pub-sub system",
		"file watcher", "log parser",
	}

	// Richer templates that reflect real coding requests with context and multiple requirements.
	templates := []string{
		"implement a %s in %s with proper error handling and unit tests",
		"create a %s class in %s that implements a %s with thread safety",
		"write a %s function in %s that handles edge cases like empty input and duplicates, then optimize it",
		"implement a %s in %s. It should support concurrent access and have O(log n) lookup. Also write benchmarks.",
		"create a %s using %s with the following requirements:\n1. Support insert, delete, and search\n2. Handle concurrent access\n3. Implement proper cleanup",
		"write a comprehensive unit test suite for a %s in %s covering happy path, error cases, and edge cases",
		"implement a function to validate and parse email addresses in %s, handling international domains and edge cases like plus addressing",
		"create a %s in %s that converts JSON to CSV, handling nested objects, arrays, and special characters correctly",
		"write a %s in %s that handles concurrent requests with proper context cancellation and graceful shutdown",
		"implement cursor-based pagination for a %s in %s, including filtering, sorting, and total count",
		"create a middleware function in %s that handles authentication, rate limiting, and request logging. Implement proper error responses.",
		"implement a producer-consumer pattern in %s with bounded channels, backpressure, and graceful shutdown. Include error handling.",
		"write a function in %s to merge k sorted arrays efficiently. Optimize for both time and space complexity.",
		"implement a cache with TTL, LRU eviction, and size limits in %s. Make it safe for concurrent use.",
		"create an iterator pattern in %s that supports filtering, mapping, and lazy evaluation over large datasets",
		"write a connection pool in %s that handles connection health checks, automatic reconnection, and configurable pool size",
		"implement binary search with variants (lower_bound, upper_bound, search_range) in %s with comprehensive tests",
		"create a retry mechanism in %s with exponential backoff, jitter, configurable max retries, and circuit breaker integration",
		"implement a %s data structure in %s optimized for memory efficiency. Compare tradeoffs vs the standard library version.",
		"write a %s abstraction in %s following clean architecture principles with dependency injection",
	}

	var prompts []string
	for _, tmpl := range templates {
		count := strings.Count(tmpl, "%s")
		lang := languages[rng.Intn(len(languages))]
		switch count {
		case 1:
			prompts = append(prompts, fmt.Sprintf(tmpl, lang))
		case 2:
			pool := structures
			if strings.Contains(tmpl, "concurrent requests") || strings.Contains(tmpl, "pagination") {
				pool = tasks
			}
			item := pool[rng.Intn(len(pool))]
			prompts = append(prompts, fmt.Sprintf(tmpl, item, lang))
		case 3:
			s := structures[rng.Intn(len(structures))]
			t := tasks[rng.Intn(len(tasks))]
			prompts = append(prompts, fmt.Sprintf(tmpl, lang, s, t))
		}
	}
	return prompts
}

func debugPrompts(rng *rand.Rand) []string {
	errors := []string{
		"nil pointer dereference", "index out of range", "deadlock",
		"race condition", "memory leak", "segfault", "stack overflow",
		"connection refused", "timeout", "permission denied",
		"undefined variable", "type mismatch", "import cycle",
		"goroutine leak", "context cancelled",
	}
	components := []string{
		"login function", "API handler", "database query", "middleware",
		"auth service", "payment processor", "email sender", "file upload",
		"websocket handler", "cron job",
	}
	stackTraces := []string{
		"\n```\ngoroutine 1 [running]:\nmain.handler(0xc000104000)\n\t/app/server.go:42\nmain.main()\n\t/app/main.go:15\n```",
		"\n```\npanic: runtime error: index out of range [5] with length 3\ngoroutine 1 [running]:\nmain.process()\n\t/app/worker.go:88\n```",
		"\n```\nError: connection refused\n  at TCPClient.connect (net.js:1141:16)\n  at Socket.connect (net.js:1200:10)\n```",
		"\n```\nTraceback (most recent call last):\n  File \"app.py\", line 42, in handle\n    result = db.execute(query)\nsqlalchemy.exc.OperationalError: connection timed out\n```",
		"\n```\nfatal error: all goroutines are asleep - deadlock!\ngoroutine 1 [chan receive]:\nmain.worker()\n\t/app/pool.go:55\n```",
	}

	// Richer templates with error context, stack traces, and multi-part asks.
	templates := []string{
		"debug this %s error in the %s. Here's the stack trace:%s\nWhat's the root cause and how do I fix it?",
		"the %s is failing with %s after the latest deploy. It works in staging but not production. Help me debug this.",
		"fix the %s in my %s. The error only happens under high load with concurrent requests. I suspect a %s.",
		"I'm getting a %s when I run the %s. Here's the error output:%s\nI've already tried restarting the service.",
		"the %s throws %s intermittently. Debug the issue and explain the root cause. Here's what I see in the logs:%s",
		"help me debug this %s issue. It started after I refactored the %s. The error is %s.",
		"why does my %s crash with %s? It only happens when processing more than 1000 items. Debug and suggest a fix.",
		"the %s is not working in production, getting %s. Here's the relevant code:\n```\nfunc handler(w http.ResponseWriter, r *http.Request) {\n  data := fetchData(r.Context())\n  process(data)\n}\n```\nDebug this.",
		"debug: %s returns %s unexpectedly. I've checked the input validation and it looks correct. What else could cause this?",
		"getting intermittent %s in the %s under load. The error happens about 1 in 100 requests. Help me find the root cause.",
		"the %s has a bug causing %s. I need to fix this urgently. Here's the error log:%s\nAnalyze and provide a fix.",
		"my %s is throwing %s after upgrading dependencies. Debug the compatibility issue and suggest how to fix it.",
		"diagnose why the %s gives %s when handling concurrent requests. I suspect a race condition in the shared state.",
		"troubleshoot the %s. It returns %s for certain inputs. Help me identify the edge case and fix the logic.",
		"the %s keeps failing with %s. I've tried the obvious fixes. Debug systematically and explain what's wrong.",
	}

	var prompts []string
	for _, tmpl := range templates {
		e := errors[rng.Intn(len(errors))]
		c := components[rng.Intn(len(components))]
		st := stackTraces[rng.Intn(len(stackTraces))]
		count := strings.Count(tmpl, "%s")
		switch count {
		case 2:
			if strings.Contains(tmpl, "debug this") || strings.Contains(tmpl, "fix the") || strings.Contains(tmpl, "diagnose") {
				prompts = append(prompts, fmt.Sprintf(tmpl, e, c))
			} else {
				prompts = append(prompts, fmt.Sprintf(tmpl, c, e))
			}
		case 3:
			if strings.Contains(tmpl, "stack trace") || strings.Contains(tmpl, "error output") || strings.Contains(tmpl, "error log") || strings.Contains(tmpl, "the logs") {
				prompts = append(prompts, fmt.Sprintf(tmpl, c, e, st))
			} else {
				e2 := errors[rng.Intn(len(errors))]
				prompts = append(prompts, fmt.Sprintf(tmpl, c, e, e2))
			}
		}
	}
	return prompts
}

func reviewPrompts(rng *rand.Rand) []string {
	techs := []string{
		"React vs Vue", "REST vs GraphQL", "SQL vs NoSQL", "Go vs Rust",
		"Docker vs Podman", "Kafka vs RabbitMQ", "Redis vs Memcached",
		"JWT vs sessions", "monolith vs microservices", "gRPC vs REST",
		"PostgreSQL vs MySQL", "TypeScript vs JavaScript",
	}
	codeSnippets := []string{
		"```go\nfunc ProcessOrder(ctx context.Context, order *Order) error {\n\tif err := validate(order); err != nil {\n\t\treturn err\n\t}\n\ttx, _ := db.Begin(ctx)\n\tdefer tx.Rollback()\n\tif err := tx.Insert(order); err != nil {\n\t\treturn err\n\t}\n\treturn tx.Commit()\n}\n```",
		"```python\ndef handle_request(request):\n    data = json.loads(request.body)\n    user = User.objects.get(id=data['user_id'])\n    result = process(user, data)\n    cache.set(f'result:{user.id}', result)\n    return JsonResponse({'status': 'ok', 'data': result})\n```",
		"```typescript\nasync function fetchUsers(filters: UserFilters): Promise<User[]> {\n  const query = buildQuery(filters);\n  const result = await db.query(query);\n  return result.rows.map(row => new User(row));\n}\n```",
		"```go\nfunc (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {\n\tconn, _ := upgrader.Upgrade(w, r, nil)\n\tdefer conn.Close()\n\tfor {\n\t\t_, msg, err := conn.ReadMessage()\n\t\tif err != nil { break }\n\t\ts.broadcast(msg)\n\t}\n}\n```",
	}

	templates := []string{
		"review this code for potential issues and suggest improvements:\n%s",
		"compare %s for our use case. We need high throughput, low latency, and good developer experience. Evaluate the tradeoffs.",
		"what are the tradeoffs between %s? Consider performance, scalability, maintenance cost, and ecosystem maturity.",
		"review the following code and evaluate it for bugs, performance issues, and security vulnerabilities:\n%s",
		"compare %s and explain when to use each. We're building a real-time analytics dashboard.",
		"code review: check this for bugs, race conditions, and error handling issues:\n%s",
		"review this code for security vulnerabilities and suggest fixes:\n%s",
		"compare %s in terms of performance, cost, and operational complexity for a team of 5 engineers.",
		"review my implementation and suggest improvements for error handling, testing, and maintainability:\n%s",
		"is %s better for our use case? We expect 10K requests/sec and need sub-100ms latency. Evaluate both options.",
		"compare %s for a startup context. Optimize for development speed, cost, and ability to scale later.",
		"review this code for scalability concerns. How would it behave under 10x load?\n%s",
		"what would you change about this implementation? Review for correctness, performance, and idiomatic style:\n%s",
		"give feedback on this design. Evaluate the error handling, resource cleanup, and concurrency safety:\n%s",
		"compare %s from a developer experience, performance, and long-term maintainability perspective.",
		"audit this code for best practices, potential bugs, and suggest improvements:\n%s",
	}

	var prompts []string
	for _, tmpl := range templates {
		if strings.Contains(tmpl, "compare") || strings.Contains(tmpl, "tradeoffs") || strings.Contains(tmpl, "better") {
			t := techs[rng.Intn(len(techs))]
			prompts = append(prompts, fmt.Sprintf(tmpl, t))
		} else {
			c := codeSnippets[rng.Intn(len(codeSnippets))]
			prompts = append(prompts, fmt.Sprintf(tmpl, c))
		}
	}
	return prompts
}

func analysisPrompts(rng *rand.Rand) []string {
	domains := []string{
		"payment processing pipeline", "user authentication flow",
		"real-time notification system", "data ingestion pipeline",
		"search indexing service", "recommendation engine",
		"order fulfillment system", "content delivery network",
		"rate limiting infrastructure", "logging and monitoring stack",
	}
	aspects := []string{
		"performance", "scalability", "reliability", "security",
		"cost efficiency", "latency", "throughput", "fault tolerance",
	}

	templates := []string{
		"analyze the %s of the %s and identify the top 3 bottlenecks. Evaluate potential optimizations and their tradeoffs.",
		"evaluate tradeoffs in the %s for %s. Compare the current approach against alternatives and recommend changes.",
		"analyze the bottlenecks in the %s. Profile the hot paths and suggest optimizations with expected impact.",
		"evaluate the %s implications of scaling the %s to 10x current load. Analyze resource requirements and failure modes.",
		"analyze the impact of migrating the %s on %s. Evaluate risks, estimate effort, and propose a migration strategy.",
		"assess the %s of our current %s against industry benchmarks. Identify gaps and prioritize improvements.",
		"analyze the failure modes in the %s. Evaluate cascading failure scenarios and suggest circuit breaker placements.",
		"evaluate whether the %s meets our %s requirements under peak load. Analyze with concrete metrics.",
		"analyze the %s architecture for potential improvements. Evaluate current design patterns and suggest alternatives with tradeoffs.",
		"assess the risks of migrating the %s to a new infrastructure. Evaluate data consistency, downtime, and rollback strategies.",
		"evaluate the cost-benefit of rewriting the %s. Analyze engineering effort vs operational savings over 12 months.",
		"analyze the %s from a %s standpoint. Compare against best practices and evaluate the gap.",
		"evaluate the monitoring gaps in the %s. Analyze observability coverage and suggest improvements for incident response.",
		"analyze the dependency chain of the %s. Evaluate single points of failure and suggest redundancy improvements.",
		"assess whether %s is adequate for the %s at our projected 3-year scale. Analyze growth patterns and capacity limits.",
		"evaluate the operational complexity of the %s. Analyze the on-call burden, common failure patterns, and suggest simplifications.",
	}

	var prompts []string
	for _, tmpl := range templates {
		count := strings.Count(tmpl, "%s")
		switch count {
		case 1:
			d := domains[rng.Intn(len(domains))]
			prompts = append(prompts, fmt.Sprintf(tmpl, d))
		case 2:
			a := aspects[rng.Intn(len(aspects))]
			d := domains[rng.Intn(len(domains))]
			prompts = append(prompts, fmt.Sprintf(tmpl, a, d))
		}
	}
	return prompts
}

func architecturePrompts(rng *rand.Rand) []string {
	systems := []string{
		"real-time chat application", "e-commerce platform",
		"video streaming service", "IoT device management platform",
		"financial trading system", "healthcare records system",
		"social media feed", "ride-sharing service",
		"content management system", "multi-tenant SaaS platform",
	}
	scales := []string{
		"10M daily active users", "1B events per day",
		"100K concurrent connections", "sub-50ms latency globally",
		"99.99% uptime SLA", "petabyte-scale data",
		"multi-region deployment", "zero-downtime deployments",
	}
	patterns := []string{
		"event sourcing", "CQRS", "saga pattern", "circuit breaker",
		"bulkhead pattern", "service mesh", "API gateway",
		"distributed caching", "write-ahead logging", "leader election",
	}

	templates := []string{
		"design a system for a %s that handles %s",
		"architect a %s at scale with %s",
		"design the data model and architecture for a %s",
		"architect a distributed %s using %s",
		"design a %s with %s requirements",
		"create a system design for a %s supporting %s",
		"architect the backend for a %s handling %s",
		"design the infrastructure for a %s with %s",
		"plan the migration architecture for a %s to support %s",
		"design a fault-tolerant %s using %s and %s",
		"architect a %s that scales horizontally with %s",
		"design the event-driven architecture for a %s",
		"create a high-level system design for a %s",
		"architect a %s with %s to ensure %s",
		"design the microservices decomposition for a %s",
		"plan the architecture for a greenfield %s at scale",
		"design the observability stack for a %s with %s",
		"architect the data pipeline for a %s handling %s",
	}

	var prompts []string
	for _, tmpl := range templates {
		count := strings.Count(tmpl, "%s")
		s := systems[rng.Intn(len(systems))]
		sc := scales[rng.Intn(len(scales))]
		p := patterns[rng.Intn(len(patterns))]
		switch count {
		case 1:
			prompts = append(prompts, fmt.Sprintf(tmpl, s))
		case 2:
			prompts = append(prompts, fmt.Sprintf(tmpl, s, sc))
		case 3:
			prompts = append(prompts, fmt.Sprintf(tmpl, s, p, sc))
		}
	}
	return prompts
}

// confirmationCases generates multi-turn conversations where the user confirms
// an assistant proposal. The assistant messages vary in complexity to test
// whether the classifier properly escalates (or not).
func confirmationCases(rng *rand.Rand) []benchmarkCase {
	type scenario struct {
		userInit  string
		assistant string
	}

	complexScenarios := []scenario{
		{
			"refactor the auth module",
			"I'll refactor the auth module. Here's my plan:\n\n1. Extract token validation into a dedicated service\n2. Implement proper error types with context\n3. Add middleware for route protection\n4. Set up refresh token rotation\n\n```go\ntype AuthService struct {\n\ttokenStore TokenStore\n\thasher     PasswordHasher\n}\n```\n\n```go\nfunc (s *AuthService) ValidateToken(ctx context.Context, token string) (*Claims, error) {\n\t// implementation\n}\n```\n\nShall I proceed?",
		},
		{
			"migrate the database to PostgreSQL",
			"Here's the migration plan:\n\nStep 1: Create PostgreSQL schema with proper indexes\nStep 2: Write data migration scripts\nStep 3: Set up dual-write during transition\nStep 4: Validate data integrity\nStep 5: Switch reads to PostgreSQL\nStep 6: Decommission old database\n\n```sql\nCREATE TABLE users (\n\tid UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n\temail TEXT UNIQUE NOT NULL\n);\n```\n\n```go\nfunc MigrateData(src *mongo.Client, dst *pgxpool.Pool) error {\n\t// batch migration logic\n}\n```\n\nReady to start?",
		},
		{
			"implement the caching layer",
			"I'll implement a multi-level caching strategy:\n\n1. L1: In-process LRU cache (hot data, ~10ms)\n2. L2: Redis cluster (warm data, ~1-5ms network)\n3. Cache invalidation via pub/sub\n\n```go\ntype CacheManager struct {\n\tl1 *lru.Cache\n\tl2 *redis.Client\n}\n```\n\n```go\nfunc (cm *CacheManager) Get(ctx context.Context, key string) ([]byte, error) {\n\tif v, ok := cm.l1.Get(key); ok {\n\t\treturn v.([]byte), nil\n\t}\n\treturn cm.l2.Get(ctx, key).Bytes()\n}\n```\n\n```go\nfunc (cm *CacheManager) Invalidate(ctx context.Context, pattern string) error {\n\t// pub/sub invalidation\n}\n```\n\nWant me to implement this?",
		},
		{
			"set up the CI/CD pipeline",
			"Here's the pipeline design:\n\nStep 1: Lint and static analysis\nStep 2: Unit tests with coverage\nStep 3: Integration tests against test containers\nStep 4: Build Docker images\nStep 5: Deploy to staging\nStep 6: Run smoke tests\nStep 7: Blue-green deploy to production\n\n```yaml\nstages:\n  - lint\n  - test\n  - build\n  - deploy-staging\n  - deploy-prod\n```\n\n```go\nfunc HealthCheck(endpoint string) error {\n\tresp, err := http.Get(endpoint + \"/health\")\n\t// validation\n}\n```\n\nShall I set this up?",
		},
		{
			"rewrite the payment processing service",
			"I'll restructure the payment service with proper error handling and idempotency:\n\n1. Implement idempotency keys for all mutations\n2. Add saga pattern for distributed transactions\n3. Integrate with Stripe webhook verification\n4. Add comprehensive audit logging\n\n```go\ntype PaymentService struct {\n\tstripe    *stripe.Client\n\tdb        *pgxpool.Pool\n\tauditLog  *AuditLogger\n}\n```\n\n```go\nfunc (ps *PaymentService) ProcessPayment(ctx context.Context, req PaymentRequest) (*PaymentResult, error) {\n\t// idempotency check\n\t// charge creation\n\t// saga coordination\n}\n```\n\n```go\nfunc (ps *PaymentService) HandleWebhook(ctx context.Context, payload []byte, sig string) error {\n\t// signature verification\n\t// event processing\n}\n```\n\nReady to implement?",
		},
	}

	moderateScenarios := []scenario{
		{
			"can you fix the login bug",
			"I found the bug. The token validation is failing because the expiry check uses UTC but the token was created with local time. I can fix this by normalizing both to UTC.\n\nWant me to make the change?",
		},
		{
			"add input validation to the form",
			"I'll add validation for email format, password strength (min 8 chars, mixed case, digit), and sanitize text inputs. Should I proceed?",
		},
		{
			"update the error messages",
			"I'll update the error messages to be more user-friendly and include error codes for debugging. The changes affect 12 files across the API handlers. Want me to go ahead?",
		},
		{
			"add pagination to the list endpoint",
			"I'll implement cursor-based pagination with a default page size of 25. The response will include `next_cursor` and `has_more` fields. Should I implement this?",
		},
		{
			"fix the sorting on the dashboard",
			"The issue is that the sort is case-sensitive. I'll normalize the sort field to lowercase before comparing. This affects the query in the dashboard handler. Want me to fix it?",
		},
	}

	simpleScenarios := []scenario{
		{
			"hello",
			"Hi! How can I help you today?",
		},
		{
			"what time is it",
			"I don't have access to real-time data, but I can help with coding questions!",
		},
		{
			"thanks for the help",
			"You're welcome! Let me know if you need anything else.",
		},
	}

	confirmations := []string{
		"do it", "yes", "go ahead", "proceed", "sure", "ok",
		"yes please", "go for it", "sounds good", "let's do it",
		"ship it", "lgtm", "approved", "implement it", "make it so",
		"yes go ahead", "please proceed", "yep", "yeah", "do that",
	}

	var cases []benchmarkCase

	// 50 complex confirmation cases
	for i := 0; i < 50; i++ {
		sc := complexScenarios[rng.Intn(len(complexScenarios))]
		conf := confirmations[rng.Intn(len(confirmations))]
		cases = append(cases, benchmarkCase{
			category: "confirmation",
			prompt:   conf,
			messages: []types.Message{
				{Role: "user", Content: rawB(sc.userInit)},
				{Role: "assistant", Content: rawB(sc.assistant)},
				{Role: "user", Content: rawB(conf)},
			},
		})
	}

	// 30 moderate confirmation cases
	for i := 0; i < 30; i++ {
		sc := moderateScenarios[rng.Intn(len(moderateScenarios))]
		conf := confirmations[rng.Intn(len(confirmations))]
		cases = append(cases, benchmarkCase{
			category: "confirmation",
			prompt:   conf,
			messages: []types.Message{
				{Role: "user", Content: rawB(sc.userInit)},
				{Role: "assistant", Content: rawB(sc.assistant)},
				{Role: "user", Content: rawB(conf)},
			},
		})
	}

	// 20 simple confirmation cases (should NOT escalate)
	for i := 0; i < 20; i++ {
		sc := simpleScenarios[rng.Intn(len(simpleScenarios))]
		conf := confirmations[rng.Intn(len(confirmations))]
		cases = append(cases, benchmarkCase{
			category: "confirmation",
			prompt:   conf,
			messages: []types.Message{
				{Role: "user", Content: rawB(sc.userInit)},
				{Role: "assistant", Content: rawB(sc.assistant)},
				{Role: "user", Content: rawB(conf)},
			},
		})
	}

	return cases
}

// --- Prompt generation ---

// pick returns n elements chosen randomly (with replacement) from pool.
func pick(rng *rand.Rand, pool []string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = pool[rng.Intn(len(pool))]
	}
	return out
}

func generatePrompts(rng *rand.Rand) []benchmarkCase {
	greetings := greetingPrompts()
	factuals := factualPrompts(rng)
	simpleCodes := simpleCodePrompts(rng)
	explanations := explanationPrompts(rng)
	codeGens := codeGenPrompts(rng)
	debugs := debugPrompts(rng)
	reviews := reviewPrompts(rng)
	analyses := analysisPrompts(rng)
	architectures := architecturePrompts(rng)

	var cases []benchmarkCase

	// Helper to build benchmarkCase from a prompt pool.
	addFromPool := func(category string, pool []string, count int) {
		selected := pick(rng, pool, count)
		for _, p := range selected {
			cases = append(cases, benchmarkCase{category: category, prompt: p})
		}
	}

	addFromPool("greetings", greetings, 100)
	addFromPool("factual", factuals, 120)
	addFromPool("simple_code", simpleCodes, 100)
	addFromPool("explanation", explanations, 100)
	addFromPool("code_gen", codeGens, 120)
	addFromPool("debug", debugs, 100)
	addFromPool("review", reviews, 80)
	addFromPool("analysis", analyses, 80)
	addFromPool("architecture", architectures, 100)

	// Confirmations are special — they have multi-turn conversations.
	cases = append(cases, confirmationCases(rng)...)

	return cases
}

// Standard tool sets that real-world coding clients send along with requests.
var ideTools = []types.Tool{
	{Type: "function", Function: types.FunctionDef{Name: "read_file", Description: "Read a file from disk"}},
	{Type: "function", Function: types.FunctionDef{Name: "write_file", Description: "Write content to a file"}},
	{Type: "function", Function: types.FunctionDef{Name: "search", Description: "Search codebase"}},
	{Type: "function", Function: types.FunctionDef{Name: "run_command", Description: "Run a shell command"}},
	{Type: "function", Function: types.FunctionDef{Name: "list_files", Description: "List files in a directory"}},
}

// buildRequest creates a ChatCompletionRequest from a benchmarkCase,
// adding realistic scaffolding (tools, system prompts, multi-turn context)
// based on the category. This mirrors how real coding tools construct requests.
func buildRequest(bc benchmarkCase) *types.ChatCompletionRequest {
	req := &types.ChatCompletionRequest{}

	// Confirmations and other cases with pre-built messages use them directly.
	if len(bc.messages) > 0 {
		req.Messages = bc.messages
		if len(bc.tools) > 0 {
			req.Tools = bc.tools
		}
		return req
	}

	switch bc.category {
	case "greetings", "factual":
		// Simple single-message requests — no tools, no system prompt.
		req.Messages = []types.Message{
			{Role: "user", Content: rawB(bc.prompt)},
		}

	case "simple_code":
		// IDE sends tools but the task itself is trivial.
		req.Messages = []types.Message{
			{Role: "user", Content: rawB(bc.prompt)},
		}
		req.Tools = ideTools

	case "explanation":
		// Sometimes has a prior turn of context.
		req.Messages = []types.Message{
			{Role: "user", Content: rawB(bc.prompt)},
		}

	case "code_gen":
		// Coding tool sends tools + system prompt.
		req.Messages = []types.Message{
			{Role: "system", Content: rawB("You are a coding assistant. Help the user write clean, idiomatic code.")},
			{Role: "user", Content: rawB(bc.prompt)},
		}
		req.Tools = ideTools

	case "debug":
		// Multi-turn: user describes problem, assistant asks for details, user shares error.
		req.Messages = []types.Message{
			{Role: "system", Content: rawB("You are a coding assistant helping debug issues.")},
			{Role: "user", Content: rawB(bc.prompt)},
		}
		req.Tools = ideTools

	case "review":
		// Code review: system prompt + tools + the review request.
		req.Messages = []types.Message{
			{Role: "system", Content: rawB("You are a senior code reviewer. Analyze code for bugs, performance issues, and best practices.")},
			{Role: "user", Content: rawB(bc.prompt)},
		}
		req.Tools = ideTools

	case "analysis":
		// Analysis: complex system prompt with reasoning keywords.
		req.Messages = []types.Message{
			{Role: "system", Content: rawB("You are a senior engineer. Analyze step-by-step and provide detailed reasoning for your conclusions.")},
			{Role: "user", Content: rawB(bc.prompt)},
		}
		req.Tools = ideTools

	case "architecture":
		// Architecture: complex system prompt, tools, detailed ask.
		req.Messages = []types.Message{
			{Role: "system", Content: rawB("You are an expert software architect. Think step-by-step about system design. Consider scalability, reliability, and performance tradeoffs in your analysis.")},
			{Role: "user", Content: rawB(bc.prompt)},
		}
		req.Tools = ideTools

	default:
		req.Messages = []types.Message{
			{Role: "user", Content: rawB(bc.prompt)},
		}
	}

	return req
}

// --- Cost model ---

type tierPricing struct {
	inputPer1M  float64
	outputPer1M float64
}

var pricing = map[types.Tier]tierPricing{
	types.Tier1: {inputPer1M: 0.80, outputPer1M: 4.00},   // Haiku
	types.Tier2: {inputPer1M: 3.00, outputPer1M: 15.00},   // Sonnet
	types.Tier3: {inputPer1M: 15.00, outputPer1M: 75.00},  // Opus
}

var baselinePricing = tierPricing{inputPer1M: 15.00, outputPer1M: 75.00} // Opus for all

const (
	avgInputTokens  = 500
	avgOutputTokens = 300
)

func costPerRequest(p tierPricing) float64 {
	return (float64(avgInputTokens) * p.inputPer1M / 1_000_000) +
		(float64(avgOutputTokens) * p.outputPer1M / 1_000_000)
}

// --- Benchmark test ---

func TestBenchmark1000Prompts(t *testing.T) {
	// 1. Initialize classifier with real config + embedded ML model.
	c := New(config.ClassifierConfig{
		Tier1Threshold: 0.35,
		Tier2Threshold: 0.70,
	})

	// 2. Generate 1000 prompts with deterministic seed.
	rng := rand.New(rand.NewSource(42))
	cases := generatePrompts(rng)

	if len(cases) != 1000 {
		t.Fatalf("expected 1000 prompts, got %d", len(cases))
	}

	// 3. Classify each prompt.
	type result struct {
		category string
		tier     types.Tier
		score    float64
		signals  map[string]float64
	}

	results := make([]result, len(cases))
	for i, bc := range cases {
		req := buildRequest(bc)
		tier, score, signals := c.Classify(req)
		results[i] = result{
			category: bc.category,
			tier:     tier,
			score:    score,
			signals:  signals,
		}
	}

	// 4. Aggregate results.

	// Overall tier distribution.
	tierCounts := map[types.Tier]int{
		types.Tier1: 0,
		types.Tier2: 0,
		types.Tier3: 0,
	}

	// Per-category breakdown.
	type categoryStats struct {
		count      int
		tier1      int
		tier2      int
		tier3      int
		totalScore float64
	}
	categories := []string{
		"greetings", "factual", "simple_code", "explanation",
		"code_gen", "debug", "review", "analysis",
		"architecture", "confirmation",
	}
	catStats := make(map[string]*categoryStats)
	for _, cat := range categories {
		catStats[cat] = &categoryStats{}
	}

	// Confirmation escalation tracking.
	confirmationTotal := 0
	confirmationEscalated := 0

	for _, r := range results {
		tierCounts[r.tier]++
		cs := catStats[r.category]
		cs.count++
		cs.totalScore += r.score
		switch r.tier {
		case types.Tier1:
			cs.tier1++
		case types.Tier2:
			cs.tier2++
		case types.Tier3:
			cs.tier3++
		}

		if r.category == "confirmation" {
			confirmationTotal++
			if r.signals["confirmation_escalation"] > 0 {
				confirmationEscalated++
			}
		}
	}

	// 5. Calculate costs.
	var aionTotalCost float64
	for tier, count := range tierCounts {
		p := pricing[tier]
		aionTotalCost += float64(count) * costPerRequest(p)
	}

	baselineTotalCost := float64(len(cases)) * costPerRequest(baselinePricing)
	savings := baselineTotalCost - aionTotalCost
	savingsPct := (savings / baselineTotalCost) * 100

	// Scale to per-1M requests.
	scaleFactor := 1_000_000.0 / float64(len(cases))
	aionPer1M := aionTotalCost * scaleFactor
	baselinePer1M := baselineTotalCost * scaleFactor
	savingsPer1M := savings * scaleFactor

	// 6. Print report.
	t.Logf("\n%s", strings.Repeat("=", 60))
	t.Logf("  AION Classifier Benchmark (1000 prompts)")
	t.Logf("%s", strings.Repeat("=", 60))

	t.Logf("\nTier Distribution:")
	t.Logf("  Tier 1 (simple):   %4d (%5.1f%%)", tierCounts[types.Tier1], pct(tierCounts[types.Tier1], len(cases)))
	t.Logf("  Tier 2 (moderate): %4d (%5.1f%%)", tierCounts[types.Tier2], pct(tierCounts[types.Tier2], len(cases)))
	t.Logf("  Tier 3 (complex):  %4d (%5.1f%%)", tierCounts[types.Tier3], pct(tierCounts[types.Tier3], len(cases)))

	t.Logf("\nCategory Breakdown:")
	t.Logf("  %-18s %5s %6s %6s %6s %9s", "Category", "Count", "Tier1", "Tier2", "Tier3", "Avg Score")
	t.Logf("  %s", strings.Repeat("-", 55))
	for _, cat := range categories {
		cs := catStats[cat]
		avg := 0.0
		if cs.count > 0 {
			avg = cs.totalScore / float64(cs.count)
		}
		t.Logf("  %-18s %5d %6d %6d %6d %9.3f", cat, cs.count, cs.tier1, cs.tier2, cs.tier3, avg)
	}

	t.Logf("\nCost Analysis (per 1M equivalent requests):")
	t.Logf("  Baseline (Opus for all):  $%.2f", baselinePer1M)
	t.Logf("  AION routed:              $%.2f", aionPer1M)
	t.Logf("  Savings:                  $%.2f (%.1f%%)", savingsPer1M, savingsPct)

	t.Logf("\nConfirmation Escalation:")
	t.Logf("  Total confirmations:      %d", confirmationTotal)
	t.Logf("  Escalated:                %d (%.1f%%)", confirmationEscalated, pct(confirmationEscalated, confirmationTotal))
	t.Logf("  Not escalated:            %d (%.1f%%)", confirmationTotal-confirmationEscalated, pct(confirmationTotal-confirmationEscalated, confirmationTotal))

	t.Logf("\n%s", strings.Repeat("=", 60))

	// Sanity checks — these are soft assertions to catch major regressions.
	if savingsPct < 20 {
		t.Errorf("expected at least 20%% savings, got %.1f%%", savingsPct)
	}
	if tierCounts[types.Tier1] == 0 {
		t.Errorf("no prompts classified as Tier 1 — classifier may be broken")
	}
	if tierCounts[types.Tier3] == 0 {
		t.Errorf("no prompts classified as Tier 3 — classifier may be broken")
	}
	if confirmationEscalated == 0 {
		t.Errorf("no confirmation escalations detected — escalation logic may be broken")
	}
}

func pct(part, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(part) / float64(total) * 100
}
