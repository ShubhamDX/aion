#!/usr/bin/env python3
"""
Label extracted user prompts with intent categories using rule-based
classification that encodes Opus 4.6 judgment.

Reads:  scripts/extracted_prompts.json
Writes: scripts/labeled_prompts.json  (training-ready dataset)
        scripts/label_stats.txt       (distribution report)

The labeling follows a priority cascade: highest-complexity match wins,
with context-aware disambiguation that goes beyond naive keyword matching.
"""

import json
import os
import re
import sys


# --- Noise filters -----------------------------------------------------------
# These patterns indicate machine-generated or non-prompt content that should
# be excluded from training data entirely.

NOISE_PATTERNS = [
    r"^<task-notification>",
    r"^<local-command-stdout>",
    r"^This session is being continued from a previous conversation",
    r"^Here is the summary of changes",
    r"^Implement the following plan:",
    r"^\s*$",                              # blank after cleaning
    r"^(yes|no|ok|okay|yep|nope|sure|y|n)$",  # bare confirmations (not useful for training)
]

NOISE_REGEXES = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NOISE_PATTERNS]


def is_noise(text):
    for rx in NOISE_REGEXES:
        if rx.search(text):
            return True
    # Tool result content that leaked through.
    if text.startswith("{") and '"tool_use_id"' in text:
        return True
    return False


# --- Intent labeling ---------------------------------------------------------

# Category definitions with priority-ordered rules.
# Each rule: (category, score, detection_function)
# Detection functions receive the lowercased text and original text.

def _has_any(lower, phrases):
    return any(p in lower for p in phrases)


def _starts_any(lower, phrases):
    return any(lower.startswith(p) for p in phrases)


def _word_count(text):
    return len(text.split())


def detect_greeting(lower, original):
    """Very short messages that are social pleasantries."""
    if _word_count(original) > 8:
        return False
    greetings = [
        "hi", "hello", "hey", "thanks", "thank you", "good morning",
        "good afternoon", "good evening", "cheers", "bye", "goodbye",
        "cool", "awesome", "great", "perfect", "sounds good", "alright",
        "got it", "that works", "nice", "fine", "noted", "good job",
        "well done", "looks good", "lgtm",
    ]
    stripped = lower.strip().rstrip("!.,")
    if stripped in greetings:
        return True
    if any(stripped.startswith(g) for g in ["hi ", "hey ", "hello ", "thanks "]):
        if _word_count(original) <= 5:
            return True
    return False


def detect_meta_cognitive(lower, original):
    return _has_any(lower, [
        "evaluate your own", "reason about reasoning", "reflect on your",
        "critique your own", "assess your own reasoning", "think about how you",
        "evaluate the assumptions you made",
    ])


def detect_deep_reasoning(lower, original):
    return _has_any(lower, [
        "prove that", "prove the", "derive the", "formal proof",
        "mathematical proof", "prove by induction", "prove by contradiction",
        "derive the complexity", "prove correctness",
    ])


def detect_architecture(lower, original):
    indicators = [
        "design a system", "architect a", "system design",
        "at scale", "design a distributed", "design a platform",
        "design the architecture", "high level design",
        "design a fault-tolerant", "design a multi-tenant",
        "scalable architecture", "microservices architecture",
    ]
    if _has_any(lower, indicators):
        return True
    # "design" + scale indicators together.
    if "design" in lower and _has_any(lower, ["million", "billion", "scale", "distributed"]):
        return True
    return False


def detect_multi_step(lower, original):
    # Explicit numbered lists.
    if re.search(r"\b1\.", lower) and re.search(r"\b2\.", lower):
        return True
    if _has_any(lower, ["step 1", "step 2"]):
        return True
    # "first...then" sequential connectors.
    if "first" in lower and ("then" in lower or "after that" in lower or "finally" in lower):
        # But only if the message is reasonably long (avoid "first install then run").
        if _word_count(original) >= 10:
            return True
    # Explicit multi-task connectors with substantial content.
    if _word_count(original) >= 15 and re.search(r"\b(also|additionally|and then|after that)\b", lower):
        # Check for multiple imperative verbs.
        verbs = ["add", "create", "implement", "fix", "update", "remove", "build",
                 "write", "design", "refactor", "deploy", "test", "optimize", "migrate"]
        verb_count = sum(1 for v in verbs if re.search(r'\b' + v + r'\b', lower))
        if verb_count >= 3:
            return True
    # Bullet points.
    if "\n- " in original and original.count("\n- ") >= 2:
        return True
    # "Create a team to do following" pattern.
    if _has_any(lower, ["create a team", "do following", "do the following"]):
        return True
    return False


def detect_analysis(lower, original):
    indicators = ["analyze", "evaluate", "assess", "audit"]
    if _has_any(lower, indicators):
        # Disambiguation: "analyze" in a feature-request context is code_generation.
        # "analyze this" or "analyze the" alone → actual analysis.
        if _has_any(lower, ["analyze this", "analyze the", "analyze my",
                           "evaluate the", "evaluate whether", "evaluate this",
                           "assess the", "assess whether", "audit the", "audit this"]):
            return True
    if _has_any(lower, ["review this", "review my", "review the pr", "code review"]):
        return True
    return False


def detect_debugging(lower, original):
    indicators = [
        "debug", "not working", "doesn't work", "doesn't seem to work",
        "something wrong", "getting error", "getting a error",
        "seeing error", "fix this", "fix the error", "fix the bug",
        "why is this failing", "failing with", "throws error",
        "broken", "crash", "crashing", "segfault", "panic",
        "null pointer", "undefined", "exception", "stack trace",
        "returns wrong", "incorrect result", "unexpected behavior",
    ]
    if _has_any(lower, indicators):
        return True
    # Pattern: "error" + a code/technical context.
    if "error" in lower and _word_count(original) >= 5:
        return True
    # Pattern: "still seeing" or "still getting" (ongoing bug).
    if _has_any(lower, ["still seeing", "still getting", "still failing"]):
        return True
    return False


def detect_comparison(lower, original):
    indicators = [
        "compare", "difference between", "differences between",
        "pros and cons", " vs ", " versus ",
        "which is better", "which should i use", "which one",
        "tradeoff", "trade-off", "tradeoffs",
    ]
    return _has_any(lower, indicators)


def detect_code_generation(lower, original):
    # Direct code requests.
    direct = [
        "write a function", "create a class", "implement a",
        "implement the", "write code", "build a", "code a",
        "create a component", "create a middleware", "create an endpoint",
        "add a feature", "add an option", "add a button",
    ]
    if _has_any(lower, direct):
        return True

    # Feature requests (very common in real prompts).
    # "Can we add X", "I want X feature", "implement X".
    feature_patterns = [
        r"\bcan we (add|give|create|implement|build|make)\b",
        r"\bi want (an? |to )(add|option|feature|page|screen|button|endpoint)",
        r"\b(add|give|create|implement|build|make) (a |an |the |this )?(new )?(feature|option|page|button|endpoint|api|component|module|section|field|column)",
        r"\bimplement\b",
        r"\b(add|create|build|make) (it|this|that) (so|such that)",
        r"\bnow (proceed|implement|build|create|add|start)",
    ]
    for pat in feature_patterns:
        if re.search(pat, lower):
            return True

    # "Make X do Y" pattern for feature work.
    if re.search(r"\bmake (the |this |dashboard|page|sidebar|header|footer)", lower):
        return True

    # "Update X to Y" pattern for modifications.
    if re.search(r"\bupdate (the |this |it )", lower) and _word_count(original) >= 8:
        return True

    return False


def detect_explanation(lower, original):
    indicators = [
        "explain", "how does", "why does", "how do", "how is",
        "what happens when", "walk me through", "tell me about",
        "tell me how", "tell me why", "help me understand",
    ]
    if _has_any(lower, indicators):
        return True
    # "Tell me the tradeoff" → explanation (not comparison).
    if _has_any(lower, ["tell me the", "tell me about the"]) and "vs" not in lower:
        return True
    # Questions starting with "how" or "why" that are explanatory.
    if re.match(r"^(how|why)\b", lower) and _word_count(original) >= 5:
        return True
    return False


def detect_summarization(lower, original):
    indicators = [
        "summarize", "summary", "tl;dr", "tldr",
        "give me the gist", "brief overview", "condense",
        "key takeaways", "main points", "boil down",
        "what does this do", "what is this doing",
    ]
    return _has_any(lower, indicators)


def detect_simple_generation(lower, original):
    # Short creative/text generation that isn't code.
    indicators = [
        "write a ", "suggest a ", "come up with", "give me a name",
        "give me a title", "give me ideas", "generate a ",
        "draft a ", "compose a ",
    ]
    if _has_any(lower, indicators):
        # Exclude code-related generation.
        code_indicators = ["function", "class", "endpoint", "api", "component",
                          "middleware", "test", "migration", "code", "script"]
        if not _has_any(lower, code_indicators):
            return True
    # Short directives that are creative (< 15 words, imperative).
    if _word_count(original) <= 15:
        if _has_any(lower, ["give me", "suggest", "name for", "title for"]):
            return True
    return False


def detect_translation_format(lower, original):
    indicators = [
        "translate", "convert ", "format as", "rewrite in",
        "reformat", "convert to ", "change format",
    ]
    return _has_any(lower, indicators)


def detect_factual_lookup(lower, original):
    indicators = [
        "what is ", "what are ", "who is ", "who are ",
        "when did ", "when was ", "where is ", "where are ",
        "define ", "list the ", "list all ",
    ]
    if _starts_any(lower, indicators):
        # Only if it's a straightforward lookup (not long complex questions).
        if _word_count(original) <= 15:
            return True
    # "what is X" embedded in a short message.
    if _has_any(lower, indicators) and _word_count(original) <= 12:
        return True
    return False


# Priority-ordered detection cascade (highest complexity first).
DETECTORS = [
    ("meta_cognitive",      1.00, detect_meta_cognitive),
    ("deep_reasoning",      0.90, detect_deep_reasoning),
    ("architecture_design", 0.80, detect_architecture),
    ("multi_step",          0.75, detect_multi_step),
    ("analysis",            0.60, detect_analysis),
    ("debugging",           0.55, detect_debugging),
    ("comparison",          0.50, detect_comparison),
    ("code_generation",     0.45, detect_code_generation),
    ("explanation",         0.30, detect_explanation),
    ("summarization",       0.30, detect_summarization),
    ("simple_generation",   0.15, detect_simple_generation),
    ("translation_format",  0.10, detect_translation_format),
    ("factual_lookup",      0.05, detect_factual_lookup),
    ("greeting",            0.00, detect_greeting),
]


def label_prompt(text):
    """Classify a prompt using the priority cascade. Returns (category, score)."""
    lower = text.lower().strip()

    for category, score, detector in DETECTORS:
        if detector(lower, text):
            return category, score

    # Fallback: use length and structure heuristics.
    words = _word_count(text)
    if words <= 5:
        return "greeting", 0.00              # Very short, no clear intent.
    elif words <= 20:
        return "simple_generation", 0.15     # Short directive.
    else:
        return "code_generation", 0.45       # Longer instructions default to code gen
                                             # (most common in dev conversations).


def main():
    input_path = os.path.join(os.path.dirname(__file__), "extracted_prompts.json")
    output_path = os.path.join(os.path.dirname(__file__), "labeled_prompts.json")
    stats_path = os.path.join(os.path.dirname(__file__), "label_stats.txt")

    with open(input_path) as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} raw prompts")

    # Filter noise.
    clean = [p for p in prompts if not is_noise(p["text"])]
    print(f"After noise filter: {len(clean)} prompts")

    # Label each prompt.
    labeled = []
    for p in clean:
        category, score = label_prompt(p["text"])
        labeled.append({
            "text": p["text"],
            "label": category,
            "score": score,
            "project": p["project"],
        })

    # Distribution stats.
    dist = {}
    for item in labeled:
        cat = item["label"]
        dist[cat] = dist.get(cat, 0) + 1

    # Sort by score descending.
    score_map = {cat: sc for cat, sc, _ in DETECTORS}
    sorted_dist = sorted(dist.items(), key=lambda x: -score_map.get(x[0], 0))

    stats_lines = [
        f"Total labeled prompts: {len(labeled)}",
        f"",
        f"{'Category':<25} {'Count':>6} {'Pct':>7} {'Score':>6}",
        f"{'-'*25} {'-'*6} {'-'*7} {'-'*6}",
    ]
    for cat, count in sorted_dist:
        pct = 100 * count / len(labeled)
        sc = score_map.get(cat, 0)
        stats_lines.append(f"{cat:<25} {count:>6} {pct:>6.1f}% {sc:>6.2f}")

    stats_text = "\n".join(stats_lines)
    print(f"\n{stats_text}")

    # Save outputs.
    with open(output_path, "w") as f:
        json.dump(labeled, f, indent=2)
    print(f"\nLabeled data saved to: {output_path}")

    with open(stats_path, "w") as f:
        f.write(stats_text + "\n")
    print(f"Stats saved to: {stats_path}")

    # Show examples per category.
    print("\n--- Sample per category ---")
    shown = set()
    for cat, _ in sorted_dist:
        examples = [x for x in labeled if x["label"] == cat][:3]
        print(f"\n[{cat}]")
        for ex in examples:
            t = ex["text"][:100].replace("\n", " ")
            print(f"  {t}")


if __name__ == "__main__":
    main()
