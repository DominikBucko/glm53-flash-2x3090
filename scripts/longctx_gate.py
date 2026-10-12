#!/usr/bin/env python3
"""Long-context retrieval and coherence gates against a running endpoint.

needles     A seeded synthetic engineering log is padded to the requested prompt length; three facts are placed at
            5%, 50% and 95% depth and the model must return exactly {"alpha": ..., "middle": ..., "omega": ...}.
            Passes when every value is present and the answer parses as that exact JSON object.
coherence   1,024 greedy output tokens on an open-ended prompt; reports the unique 4-gram ratio of the output
            (degenerate loops fall far below 0.9).

The haystack is generated from fixed word lists with a fixed seed: no external or private data. Each request
carries a fresh cache_salt so prefix caching cannot shortcut prefill.

  scripts/longctx_gate.py --lengths 32768 131072 260000 --coherence --output gates.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import secrets
import sys
import time
import urllib.request
from pathlib import Path

NEEDLES = {"alpha": "Kestrel-271828", "middle": "Citrine-314159", "omega": "Juniper-161803"}
SUBJECTS = ["The scheduler", "The cache planner", "A replica", "The gateway", "The indexer", "The batch worker",
            "The lease manager", "A canary host", "The tokenizer service", "The expert router", "The ingest job"]
VERBS = ["retried", "flushed", "compacted", "rebalanced", "evicted", "resumed", "throttled", "rotated",
         "checkpointed", "drained", "promoted", "skipped"]
OBJECTS = ["shard 7", "the warm pool", "two stale leases", "the write-ahead log", "a cold partition",
           "the routing table", "pending callbacks", "the snapshot queue", "its backlog", "the retry budget"]
TAILS = ["after a 40 ms stall.", "without data loss.", "while the queue stayed below 30%.",
         "and logged a warning.", "during the nightly window.", "before the deadline.",
         "on the second attempt.", "with no client-visible effect."]


def sentence(rng: random.Random, i: int) -> str:
    return f"[{i:06d}] {rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}"


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def count_tokens(base: str, model: str, text: str, timeout: float) -> int:
    return post(f"{base}/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}, timeout)["count"]


def build_prompt(base: str, model: str, target: int, timeout: float) -> tuple[str, int]:
    rng = random.Random(20261012)
    lines = [sentence(rng, i) for i in range(target // 12 + 64)]
    per_line = count_tokens(base, model, "\n".join(lines[:400]), timeout) / 400

    def assemble(n: int) -> str:
        body = lines[:n]
        for frac, key in ((0.05, "alpha"), (0.50, "middle"), (0.95, "omega")):
            body.insert(int(len(body) * frac), f"NOTE: the {key} access code is {NEEDLES[key]}.")
        return ("Below is an operations log. Three NOTE lines in it state access codes.\n\n" + "\n".join(body) +
                '\n\nReturn only a JSON object {"alpha": ..., "middle": ..., "omega": ...} with the three access '
                "codes from the NOTE lines, nothing else.")

    n = int((target - 120) / per_line)
    for _ in range(6):
        prompt = assemble(n)
        got = count_tokens(base, model, prompt, timeout)
        if abs(got - target) <= 64:
            break
        n = max(1, int(n * target / got))
    return prompt, got


def chat(base: str, model: str, content: str, max_tokens: int, timeout: float, effort: str) -> tuple[dict, float]:
    started = time.perf_counter()
    result = post(f"{base}/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
        "temperature": 0, "cache_salt": secrets.token_hex(16),
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort},
    }, timeout)
    return result, time.perf_counter() - started


def needles(base: str, model: str, length: int, timeout: float) -> dict:
    prompt, tokens = build_prompt(base, model, length, timeout)
    result, seconds = chat(base, model, prompt, 512, timeout, "low")
    answer = (result["choices"][0]["message"].get("content") or "").strip()
    hits = {k: v in answer for k, v in NEEDLES.items()}
    match = re.search(r"\{.*\}", answer, re.S)
    try:
        exact = json.loads(match.group(0)) == NEEDLES if match else False
    except json.JSONDecodeError:
        exact = False
    return {"gate": "needles", "prompt_tokens": result["usage"]["prompt_tokens"], "requested": length,
            "built_tokens": tokens, "hits": hits, "all_needles": all(hits.values()), "exact_json": exact,
            "seconds": round(seconds, 1), "answer": answer[:300]}


def coherence(base: str, model: str, timeout: float) -> dict:
    prompt = ("Write a long, detailed technical essay on how operating systems schedule threads across CPU "
              "cores, covering fairness, cache affinity, NUMA, and real-time constraints.")
    result, seconds = chat(base, model, prompt, 1024, timeout, "low")
    msg = result["choices"][0]["message"]
    text = (msg.get("reasoning") or "") + (msg.get("content") or "")
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    ratio = len(set(grams)) / max(1, len(grams))
    return {"gate": "coherence", "completion_tokens": result["usage"]["completion_tokens"],
            "unique_4gram_ratio": round(ratio, 4), "pass": ratio >= 0.9, "seconds": round(seconds, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--lengths", type=int, nargs="*", default=[32768, 131072, 260000])
    ap.add_argument("--coherence", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    results = []
    for length in args.lengths:
        results.append(needles(base, args.model, length, args.timeout))
        print(json.dumps(results[-1]), flush=True)
    if args.coherence:
        results.append(coherence(base, args.model, args.timeout))
        print(json.dumps(results[-1]), flush=True)
    ok = all(r.get("all_needles", True) and r.get("exact_json", True) and r.get("pass", True) for r in results)
    if args.output:
        args.output.write_text(json.dumps({"schema_version": 1, "pass": ok, "results": results}, indent=1) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
