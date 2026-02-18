#!/usr/bin/env python3
"""
Train the DEFAULT intent classifier from:
  1. Public datasets (Databricks Dolly 15K — CC-BY-SA-3.0)
  2. Anonymized real prompts (if available)
  3. Synthetic examples

This model is safe for public repos — no private terms in the vocabulary.

Output: models/intent_classifier_default.json (committed to repo)
"""

import json
import os
import re
import sys
import urllib.request
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANON_PATH = os.path.join(SCRIPT_DIR, "anonymized_prompts.json")
DOLLY_CACHE = os.path.join(SCRIPT_DIR, "dolly_15k.jsonl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "models", "intent_classifier_default.json")

DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

CATEGORY_SCORES = {
    "greeting": 0.00, "factual_lookup": 0.05, "translation_format": 0.10,
    "simple_generation": 0.15, "explanation": 0.30, "summarization": 0.30,
    "code_generation": 0.45, "comparison": 0.50, "debugging": 0.55,
    "analysis": 0.60, "multi_step": 0.75, "architecture_design": 0.80,
    "deep_reasoning": 0.90, "meta_cognitive": 1.00,
}

# Comprehensive synthetic data — generic, no private terms.
SYNTHETIC = {
    "greeting": [
        "hi", "hello", "hey", "hey there", "hi there",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "thank you so much", "thanks a lot",
        "bye", "goodbye", "see you later", "have a nice day",
        "how are you", "how are you doing", "what's up",
        "howdy", "greetings", "yo", "sup",
        "cheers", "take care", "nice to meet you",
        "ok thanks", "great thanks", "perfect thank you",
        "cool", "awesome", "got it", "ok", "okay",
        "sounds good", "that works", "fine", "alright",
        "no problem", "no worries", "all good",
        "appreciate it", "much appreciated",
        "noted", "sure", "yep", "absolutely",
        "good job", "well done", "nice work", "looks good",
    ],
    "factual_lookup": [
        "what is photosynthesis", "who invented the telephone",
        "when did world war 2 end", "define polymorphism",
        "list the planets in our solar system", "what is the capital of Japan",
        "who wrote hamlet", "what is the speed of light",
        "what is the boiling point of water", "define recursion",
        "what is a prime number", "what does HTTP stand for",
        "who founded Google", "what is TCP",
        "what does REST stand for", "who created Linux",
        "what is the OSI model", "define idempotent",
        "what is a hash table", "what are the SOLID principles",
        "what is an API", "where is mount everest",
        "what is the largest ocean", "when was python created",
        "what is the currency of Japan", "what year did bitcoin launch",
        "who painted the mona lisa", "list all continents",
        "what is the chemical formula for water",
        "where is the config file", "what is the path to migrations",
        "what port is the server running on",
        "what version of node are we using",
        "where are the logs stored",
        "what is the default timeout",
        "what database engine are we using",
        "where is the main entry point",
        "what is the license for this library",
        "how many endpoints do we have",
        "what environment variables are required",
    ],
    "translation_format": [
        "translate this to Spanish", "convert this JSON to YAML",
        "format this as a markdown table", "rewrite this in formal English",
        "convert this XML to JSON", "translate this paragraph to German",
        "rewrite this email in a professional tone",
        "convert this timestamp to UTC", "format this data as CSV",
        "rewrite this in simpler words", "convert inches to centimeters",
        "translate this code from Python to JavaScript",
        "convert this hex color to RGB", "convert camelCase to snake_case",
        "format this output as a table",
        "convert this curl command to Python requests",
        "rewrite this function signature in TypeScript",
        "convert binary to decimal", "translate this to Mandarin",
        "convert this YAML config to TOML", "reformat this date to ISO 8601",
        "translate this regex to plain English",
        "rewrite this commit message to be clearer",
        "format this JSON with proper indentation",
        "convert this SQL query to MongoDB",
        "rewrite this paragraph to be more concise",
        "translate this pseudocode to actual code",
        "convert these temperatures from Celsius to Fahrenheit",
        "reformat this code with proper indentation",
        "format this log output for readability",
        "convert this REST API to GraphQL",
        "rewrite this in active voice",
        "format these results as a bulleted list",
        "convert this callback-based code to async await",
        "translate this Go code to Rust",
    ],
    "simple_generation": [
        "write a haiku about spring", "give me a name for my cat",
        "suggest a title for my blog post", "come up with a slogan for our team",
        "write a short bio for my profile", "generate a random password",
        "give me five ideas for a birthday gift",
        "suggest some variable names for a user service",
        "write a one-liner joke", "come up with a project name",
        "write a tweet about our product launch",
        "suggest an email subject line", "give me a catchy tagline",
        "write a short description for this app",
        "suggest a color scheme for a dashboard",
        "come up with a name for this function",
        "write a placeholder text for the homepage",
        "give me some sample data for testing",
        "suggest three database names", "write a commit message for this change",
        "give me a metaphor for technical debt",
        "suggest a greeting message for the app",
        "write a brief changelog entry",
        "come up with an acronym for our service",
        "give me a few enum values for status types",
        "suggest some test case names", "write a one-sentence summary",
        "generate a UUID", "suggest a domain name",
        "write a regex for email validation",
        "give me a cron expression for every Monday",
        "suggest some HTTP status codes for this endpoint",
        "come up with realistic test data for users",
        "write an alt text for this image",
        "suggest a logging format",
    ],
    "explanation": [
        "explain how DNS works", "how does a blockchain work",
        "why does ice float on water", "how do neural networks learn",
        "explain the concept of dependency injection",
        "how does garbage collection work in Go",
        "why is HTTPS more secure than HTTP",
        "explain how a load balancer distributes traffic",
        "how does a hash map work internally", "explain the CAP theorem",
        "why do we need database indexes",
        "how does TCP ensure reliable delivery",
        "explain eventual consistency", "how does OAuth 2.0 work",
        "explain how Docker containers differ from virtual machines",
        "how does a compiler optimize code",
        "explain the difference between concurrency and parallelism",
        "how does a CDN improve performance", "why do race conditions happen",
        "explain how public key cryptography works",
        "how does git rebase differ from merge",
        "explain the event loop in JavaScript",
        "how does a B-tree index work",
        "why is normalization important in databases",
        "explain how Kubernetes schedules pods",
        "how does rate limiting work", "explain the observer pattern",
        "how does a message queue work",
        "why does recursion use more memory than iteration",
        "explain how WebSockets maintain a connection",
        "how does a reverse proxy work",
        "explain the ACID properties of transactions",
        "how do microservices communicate",
        "why is connection pooling important",
        "explain how TLS handshake works",
    ],
    "summarization": [
        "summarize this article", "give me the tl;dr",
        "what's the gist of this paper", "brief overview of this document",
        "summarize the key points",
        "can you condense this into a few bullet points",
        "give me a quick summary of this meeting transcript",
        "summarize what happened in this pull request",
        "what are the main takeaways from this report",
        "tl;dr this thread", "give me the highlights",
        "summarize this long email chain",
        "what's the short version of this", "condense this changelog",
        "summarize the discussion in this issue",
        "give me the executive summary", "sum up the main arguments",
        "what are the key findings",
        "reduce this to the essential points",
        "summarize this research paper abstract",
        "give a brief recap of this conversation",
        "what's the bottom line here",
        "distill the main ideas from this chapter",
        "summarize this RFC", "give me a one-paragraph summary",
        "what is this code file doing in summary",
        "boil this down to the essentials",
        "summarize the pros and cons discussed",
        "overview of this specification",
        "brief summary of the architecture doc",
    ],
    "code_generation": [
        "write a function to reverse a string",
        "create a class for a shopping cart",
        "implement a binary search algorithm",
        "write code to parse a CSV file", "implement a LRU cache",
        "write a REST API endpoint for user login",
        "create a middleware for authentication",
        "implement a rate limiter in Go",
        "write a Python script to scrape a webpage",
        "create a database migration for adding a users table",
        "implement merge sort", "write a unit test for this function",
        "create a Dockerfile for this application",
        "implement a WebSocket server",
        "write a function to validate email addresses",
        "create an interface for a payment processor",
        "implement connection pooling",
        "write a CLI tool that accepts flags",
        "create a retry mechanism with exponential backoff",
        "implement a pub/sub pattern",
        "write a function to flatten a nested JSON object",
        "create a React component for a dropdown menu",
        "implement pagination for this API",
        "write a worker pool in Go",
        "create a custom error type with stack traces",
        "implement a trie data structure",
        "write a function to debounce API calls",
        "create a health check endpoint",
        "implement a circuit breaker",
        "write a stored procedure for updating balances",
        "build a simple HTTP server",
        "implement a token bucket algorithm",
        "create a logger that writes to both file and stdout",
        "write a GraphQL resolver",
        "add a button to delete the selected item",
        "can we add an option to upload files",
        "implement the export to PDF feature",
        "add pagination to the list view",
        "create an endpoint for bulk operations",
    ],
    "comparison": [
        "compare React and Vue",
        "what's the difference between TCP and UDP",
        "pros and cons of microservices",
        "Python vs JavaScript for web development",
        "compare SQL and NoSQL databases",
        "REST vs GraphQL which should I use",
        "difference between mutex and semaphore",
        "pros and cons of monorepo vs polyrepo",
        "compare Docker and Podman",
        "gRPC vs REST for microservices",
        "what's the difference between threads and goroutines",
        "compare PostgreSQL and MySQL",
        "Kubernetes vs Docker Swarm",
        "pros and cons of serverless",
        "compare Redis and Memcached",
        "static typing vs dynamic typing which is better",
        "difference between integration tests and unit tests",
        "compare monolith and microservices architectures",
        "AWS vs GCP vs Azure for startups",
        "pros and cons of using an ORM",
        "compare Kafka and RabbitMQ",
        "JWT vs session cookies which is more secure",
        "compare Go and Rust for systems programming",
        "difference between pub/sub and message queue",
        "compare horizontal and vertical scaling",
        "ACID vs BASE consistency models",
        "compare WebSockets and Server-Sent Events",
        "pros and cons of event sourcing",
        "difference between composition and inheritance",
        "tell me the tradeoff between approach A and B",
    ],
    "debugging": [
        "debug this code it's not printing the right output",
        "fix this error in my program", "why is this not working",
        "I'm getting a null pointer exception",
        "this function returns wrong results",
        "help me find the bug in this code",
        "why am I getting a segmentation fault",
        "my API returns 500 internal server error",
        "this query is returning empty results",
        "fix the memory leak in this function",
        "debug why this test is failing",
        "why does this crash when I pass nil",
        "this loop never terminates what's wrong",
        "getting a deadlock in my concurrent code",
        "fix this race condition",
        "my Docker container keeps restarting",
        "why is this returning undefined instead of the value",
        "this regex isn't matching what I expect",
        "debug this SQL query it's too slow",
        "fix the CORS error I'm getting",
        "why is my state not updating in React",
        "this throws a timeout error after 30 seconds",
        "fix the off-by-one error in this loop",
        "my goroutine is leaking",
        "debug why this middleware isn't being called",
        "fix the broken pipe error in my connection handler",
        "why does this work locally but fail in production",
        "getting permission denied when writing to this path",
        "this JSON unmarshaling silently fails",
        "fix this stack overflow in the recursive function",
        "still seeing error on the dashboard page",
        "something wrong with the calculation logic",
        "the import is not working for PDF files",
        "UI shows 0 in all values after the update",
        "on UI I'm getting incorrect totals",
    ],
    "analysis": [
        "analyze this dataset for trends",
        "evaluate the performance of this algorithm",
        "assess the security of this code",
        "review my architecture decisions",
        "analyze the time complexity of this function",
        "evaluate whether this design pattern is appropriate",
        "assess the scalability of this approach",
        "review this pull request for issues",
        "analyze the bottlenecks in this system",
        "evaluate the tradeoffs of this caching strategy",
        "assess the test coverage of this module",
        "review this database schema for normalization issues",
        "analyze the error patterns in these logs",
        "evaluate the readability of this code",
        "assess the risk of this migration",
        "review the API design for consistency",
        "analyze memory usage of this service",
        "assess the maintainability of this codebase",
        "review this security configuration",
        "analyze the impact of adding this index",
        "evaluate the cost implications of this architecture",
        "review this error handling strategy",
        "analyze the latency distribution of these requests",
        "evaluate whether we should adopt this library",
        "assess the performance of this query plan",
        "review this deployment pipeline for gaps",
        "analyze the dependency graph of this project",
        "evaluate the observability of this system",
        "assess the reliability of this failover mechanism",
        "give me comprehensive tech stack being used",
    ],
    "multi_step": [
        "first gather the requirements then design the database and finally implement the API",
        "step 1 set up the environment step 2 install dependencies step 3 run the tests",
        "1. create the schema 2. write migrations 3. seed the database",
        "first analyze the current code then refactor it and then add tests",
        "1. set up the project structure 2. implement the core logic 3. add error handling 4. write tests",
        "first create the interface then implement it for each provider",
        "step 1 parse the input step 2 transform the data step 3 write the output",
        "1. design the API 2. implement endpoints 3. add authentication 4. deploy",
        "start by profiling the code then identify bottlenecks and finally optimize them",
        "first write the unit tests then implement the feature to make them pass",
        "step 1 back up the database step 2 run the migration step 3 verify the data",
        "1. audit the current security 2. identify vulnerabilities 3. implement fixes 4. retest",
        "first extract the shared logic then create an interface and finally refactor both callers",
        "1. read the existing tests 2. identify gaps 3. write missing tests",
        "first set up logging then add metrics collection then create dashboards",
        "1. break down the monolith 2. extract microservices 3. set up communication",
        "first create feature branch then implement changes then open PR then merge",
        "step 1 analyze the requirements step 2 design the solution step 3 implement step 4 test",
        "1. plan the migration 2. write scripts 3. test on staging 4. execute on production",
        "now proceed with remaining implementation for the MVP",
        "create a team to do the following tasks",
        "explore the project and then implement the missing features",
        "read the config first then update it and restart the server",
        "finish everything that's pending",
    ],
    "architecture_design": [
        "design a system for handling millions of concurrent users",
        "architect a microservices platform for e-commerce",
        "design a distributed caching system that scales horizontally",
        "how would you build a real-time notification system at scale",
        "design a system for processing 10 million events per second",
        "architect a multi-tenant SaaS platform",
        "design a fault-tolerant payment processing system",
        "how would you architect a global CDN from scratch",
        "design a system for real-time collaborative editing",
        "architect a data pipeline that handles petabytes of data",
        "design a URL shortener that serves billions of redirects",
        "how would you design Twitter's timeline service at scale",
        "architect an API gateway with rate limiting and authentication",
        "design a distributed task queue system",
        "how would you build a search engine like Elasticsearch",
        "design a system for handling file uploads at scale",
        "architect a monitoring and alerting platform",
        "design a database sharding strategy for a social network",
        "how would you design a ride-sharing system like Uber",
        "architect a CI/CD platform for thousands of microservices",
        "design a system for A/B testing at scale",
        "how would you architect a video streaming platform",
        "design a distributed configuration management system",
        "architect a zero-downtime deployment system",
        "design a system that provides exactly-once message delivery",
        "how would you design a distributed rate limiter",
        "architect a multi-region active-active database setup",
        "design a recommendation engine for an e-commerce platform",
        "how would you build a log aggregation system at scale",
        "design a system for managing feature flags across microservices",
    ],
    "deep_reasoning": [
        "prove that the square root of 2 is irrational",
        "derive the time complexity of this recursive algorithm",
        "provide a formal proof that P != NP implies certain problems are intractable",
        "prove that this algorithm is correct using loop invariants",
        "derive the amortized complexity of dynamic array insertions",
        "prove that this sorting algorithm is stable",
        "derive the recurrence relation for this divide and conquer algorithm",
        "prove by induction that this recursive formula is correct",
        "derive the expected time complexity of quicksort",
        "prove that this graph algorithm terminates for all inputs",
        "derive the lower bound for comparison-based sorting",
        "prove the correctness of this distributed consensus algorithm",
        "derive the space complexity of this memoization approach",
        "prove that these two regular expressions are equivalent",
        "derive the probability of hash collision with this table size",
        "prove that this greedy algorithm produces an optimal solution",
        "derive the master theorem for this recurrence",
        "prove that this lock-free data structure is linearizable",
        "derive the throughput formula for this queuing model",
        "prove by contradiction that this problem has no polynomial solution",
        "derive the information-theoretic lower bound for this problem",
        "prove the safety property of this concurrent protocol",
        "derive the optimal batch size given these constraints",
        "prove that this type system is sound",
        "provide a mathematical proof that this hash function is uniform",
    ],
    "meta_cognitive": [
        "evaluate your own reasoning about this problem",
        "reason about your reasoning process here",
        "reflect on your approach to solving this",
        "critique your own answer and identify weaknesses",
        "think about how you arrived at this conclusion and whether it's sound",
        "evaluate the assumptions you made in your analysis",
        "reflect on whether your reasoning was biased",
        "assess the confidence level of your own answer",
        "reason about what you might be getting wrong",
        "evaluate your own understanding of this topic",
        "reflect on the limitations of your approach",
        "think about alternative perspectives you might have missed",
        "critique your own code review and identify blind spots",
        "reason about the reasoning behind your design choice",
        "evaluate whether your explanation was complete and accurate",
        "reflect on how you could improve your analysis",
        "assess whether your reasoning followed a logical structure",
        "think about the weaknesses in your argument",
        "evaluate your own problem-solving strategy",
        "reason about what information you lack to give a better answer",
    ],
}


# ---------------------------------------------------------------------------
# Dolly 15K download + intent labeling
# ---------------------------------------------------------------------------
# We re-label Dolly prompts using the same detection cascade as
# label_prompts.py so categories match our 14-class schema exactly.

def _has_any(lower, phrases):
    return any(p in lower for p in phrases)

def _word_count(text):
    return len(text.split())

def _label_dolly_prompt(text):
    """Label a single prompt using our intent detection cascade."""
    lower = text.lower().strip()
    words = _word_count(text)

    # Meta-cognitive
    if _has_any(lower, ["evaluate your own", "reason about reasoning",
                        "reflect on your", "critique your own"]):
        return "meta_cognitive"

    # Deep reasoning
    if _has_any(lower, ["prove that", "prove the", "derive the",
                        "formal proof", "mathematical proof"]):
        return "deep_reasoning"

    # Architecture/design
    if _has_any(lower, ["design a system", "architect a", "system design",
                        "at scale", "design a distributed"]):
        return "architecture_design"
    if "design" in lower and _has_any(lower, ["million", "billion", "scale", "distributed"]):
        return "architecture_design"

    # Multi-step
    if re.search(r"\b1\.", lower) and re.search(r"\b2\.", lower):
        return "multi_step"
    if _has_any(lower, ["step 1", "step 2"]):
        return "multi_step"
    if "first" in lower and ("then" in lower or "finally" in lower) and words >= 10:
        return "multi_step"

    # Analysis
    if _has_any(lower, ["analyze this", "analyze the", "evaluate the",
                        "evaluate whether", "assess the", "audit the",
                        "review this", "review my"]):
        return "analysis"

    # Debugging
    if _has_any(lower, ["debug", "not working", "doesn't work", "getting error",
                        "fix this", "fix the error", "fix the bug", "broken",
                        "crash", "exception", "error"]):
        if words >= 5:
            return "debugging"

    # Comparison
    if _has_any(lower, ["compare", "difference between", "pros and cons",
                        " vs ", " versus ", "which is better"]):
        return "comparison"

    # Code generation
    if _has_any(lower, ["write a function", "create a class", "implement a",
                        "implement the", "write code", "build a"]):
        return "code_generation"
    if re.search(r"\bimplement\b", lower):
        return "code_generation"

    # Explanation
    if _has_any(lower, ["explain", "how does", "why does", "how do",
                        "walk me through", "help me understand"]):
        return "explanation"
    if re.match(r"^(how|why)\b", lower) and words >= 5:
        return "explanation"

    # Summarization
    if _has_any(lower, ["summarize", "summary", "tl;dr", "tldr",
                        "give me the gist", "brief overview", "condense"]):
        return "summarization"

    # Simple generation
    if _has_any(lower, ["write a ", "suggest a ", "come up with",
                        "give me a name", "generate a ", "draft a "]):
        code_terms = ["function", "class", "endpoint", "api", "component",
                      "middleware", "test", "migration", "code", "script"]
        if not _has_any(lower, code_terms):
            return "simple_generation"

    # Translation/format
    if _has_any(lower, ["translate", "convert ", "format as", "rewrite in",
                        "reformat", "convert to "]):
        return "translation_format"

    # Factual lookup
    if re.match(r"^(what is |what are |who is |when did |when was |where is |define |list the )", lower):
        if words <= 15:
            return "factual_lookup"

    # Greeting
    greetings = ["hi", "hello", "hey", "thanks", "thank you", "cool",
                 "awesome", "great", "ok", "okay", "bye", "goodbye"]
    stripped = lower.strip().rstrip("!.,")
    if stripped in greetings and words <= 5:
        return "greeting"

    # Fallback based on length.
    if words <= 5:
        return "greeting"
    elif words <= 20:
        return "simple_generation"
    else:
        return "code_generation"


def download_dolly():
    """Download Dolly 15K dataset, return list of (text, label) tuples."""
    # Use cached file if available.
    if os.path.exists(DOLLY_CACHE):
        print(f"Loading cached Dolly data from {DOLLY_CACHE}")
        with open(DOLLY_CACHE) as f:
            lines = f.readlines()
    else:
        print(f"Downloading Dolly 15K from HuggingFace...")
        try:
            req = urllib.request.Request(DOLLY_URL, headers={"User-Agent": "AION-Trainer/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            # Cache locally.
            with open(DOLLY_CACHE, "w") as f:
                f.write(raw)
            lines = raw.strip().split("\n")
            print(f"  Downloaded {len(lines)} records, cached to {DOLLY_CACHE}")
        except Exception as e:
            print(f"  WARNING: Could not download Dolly dataset: {e}")
            print(f"  Continuing without public data.")
            return []

    results = []
    skipped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        instruction = record.get("instruction", "").strip()
        if not instruction or len(instruction) < 5:
            skipped += 1
            continue

        # Label using our 14-category cascade.
        label = _label_dolly_prompt(instruction)
        results.append((instruction, label))

    if skipped:
        print(f"  Skipped {skipped} invalid/empty Dolly records")
    return results


def to_native(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    texts, labels = [], []

    # 1. Download and label Dolly 15K (public dataset).
    dolly_data = download_dolly()
    if dolly_data:
        for text, label in dolly_data:
            texts.append(text)
            labels.append(label)
        dolly_dist = Counter(l for _, l in dolly_data)
        print(f"Dolly: {len(dolly_data)} prompts labeled across {len(dolly_dist)} categories")
        for cat, count in dolly_dist.most_common():
            print(f"  {cat}: {count}")

    # 2. Load anonymized real prompts if available.
    if os.path.exists(ANON_PATH):
        with open(ANON_PATH) as f:
            anon = json.load(f)
        for p in anon:
            texts.append(p["text"])
            labels.append(p["label"])
        print(f"\nLoaded {len(anon)} anonymized real prompts")
    else:
        print("\nNo anonymized prompts found — using public + synthetic data only")

    # 3. Add all synthetic data.
    for cat, examples in SYNTHETIC.items():
        for ex in examples:
            texts.append(ex)
            labels.append(cat)

    final_counts = Counter(labels)
    print(f"\nFinal dataset: {len(texts)} samples, {len(final_counts)} classes")
    for cat, count in final_counts.most_common():
        print(f"  {cat}: {count}")

    # 3. Train.
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
        max_features=5000,
        sublinear_tf=True,
        norm="l2",
    )
    X = vectorizer.fit_transform(texts)
    print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")

    model = LogisticRegression(
        max_iter=1000, solver="lbfgs", C=1.0, class_weight="balanced",
    )
    model.fit(X, labels)
    print(f"Training accuracy: {model.score(X, labels):.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, labels, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    y_pred = model.predict(X)
    print("\n" + classification_report(labels, y_pred, zero_division=0))

    # 4. Verify vocabulary is clean — no private terms.
    private_terms = list(PROJECT_REPLACEMENTS.keys()) + list(VENDOR_REPLACEMENTS.keys()) + list(NAME_REPLACEMENTS.keys())
    leaked = [t for t in private_terms if t in vectorizer.vocabulary_]
    if leaked:
        print(f"WARNING: Private terms found in vocabulary: {leaked}")
    else:
        print("Vocabulary is clean — no private terms detected")

    # 5. Export.
    export = to_native({
        "vocabulary": dict(vectorizer.vocabulary_),
        "idf_weights": vectorizer.idf_,
        "model_weights": model.coef_,
        "model_intercepts": model.intercept_,
        "categories": list(model.classes_),
        "category_scores": {c: CATEGORY_SCORES[c] for c in model.classes_},
        "config": {"sublinear_tf": True, "norm": "l2", "token_pattern": r"(?u)\b\w\w+\b"},
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(export, f)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nExported default model to {OUTPUT_PATH} ({size_kb:.1f} KB)")


# Import private term lists from anonymize script for vocabulary check.
from anonymize_prompts import PROJECT_REPLACEMENTS, VENDOR_REPLACEMENTS, NAME_REPLACEMENTS

if __name__ == "__main__":
    main()
