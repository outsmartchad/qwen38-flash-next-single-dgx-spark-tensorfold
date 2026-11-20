#!/usr/bin/env python3
"""Measure streaming completion and long-prompt prefill on the loopback server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading
import time

from local_api import ApiError, CONTEXT, DEFAULT_PORT, HOST, MODEL
from local_api import json_request, port_number, request, require_model, timed_connection

PROFILES = {
    "quick": ((1, 5), (4096,)),
    "standard": ((1, 3, 5), (4096, 32768, 131072)),
    "long": ((1, 3, 5), (4096, 32768, 131072, 200000)),
    "full-decode": ((1, 2, 3, 4, 5), ()),
    "full-prefill": ((), (4096, 8192, 16384, 32768, 65536, 131072, 200000)),
}
TEMPLATE = {"enable_thinking": False}
COMPLETION_BUDGET = 256
STREAM_LINE_LIMIT = 256 * 1024
REGIONS = ("cedar", "bay", "canyon", "delta", "field", "grove", "hill", "lake")
PROSE_TOPICS = ("hash tables", "B-trees", "ring buffers", "skip lists", "Bloom filters")
CODE_TOPICS = ("binary heap", "ring buffer", "interval index", "prefix trie", "LRU cache")


def messages(text):
    return [{"role": "user", "content": text}]


def token_count(port, text, timeout):
    reply = json_request(port, "/tokenize", {
        "model": MODEL, "messages": messages(text), "chat_template_kwargs": TEMPLATE,
        "add_generation_prompt": True,
    }, timeout=timeout)
    count = reply.get("count")
    if type(count) is not int or not 0 < count <= CONTEXT:
        raise ApiError("/tokenize did not return a valid count for the configured window")
    return count


def output_prompt(seed, stream, workload="inventory"):
    run_id = hashlib.sha256(f"{seed}:{stream}".encode()).hexdigest()[:16]
    if workload == "prose":
        topic = PROSE_TOPICS[stream]
        return (f"Batch {run_id}. Write a long tutorial about {topic}. Explain the design, "
                "operations, trade-offs, memory layout, and failure cases. Use plain prose "
                "with detailed examples and continue for at least 800 words.")
    if workload == "code":
        topic = CODE_TOPICS[stream]
        return (f"Batch {run_id}. Write a complete Python implementation of a {topic}. "
                "Include the main data structure, helpers, boundary handling, and at least "
                "eight unittest cases. Output only Python source.")
    return (f"Batch {run_id}. Generate at least 75 lines of synthetic inventory records. "
            "Each line contains an item number, a tool name, and a two-word description. "
            "Start with item 1, increment by one, and write the records with no heading.")


def long_prompt(seed, rows):
    rng = random.Random(seed)
    batch = rng.getrandbits(64)
    marker = f"FROST-{rng.getrandbits(32):08X}"
    records = [f"Batch {batch:016x}. These records are synthetic.",
               "Find the audit code and reply with the code only."]
    for index in range(rows):
        if index == rows // 2:
            records.append(f"AUDIT CODE: {marker}.")
        records.append(f"record {index:05d} | hub {rng.choice(REGIONS)} "
                       f"| units {rng.randrange(100, 1000)} | lot {rng.getrandbits(32):08x}")
    records.append("What is the audit code? Reply with the code only.")
    return "\n".join(records), marker


def size_prompt(port, seed, target, timeout):
    sample, _ = long_prompt(seed, 32)
    tokens_per_row = token_count(port, sample, timeout) / 32
    rows = max(1, round(target / tokens_per_row))
    for _ in range(3):
        prompt, marker = long_prompt(seed, rows)
        count = token_count(port, prompt, timeout)
        if abs(count - target) <= target * 0.015:
            return prompt, marker, count
        rows = max(1, round(rows * target / count))
    raise ApiError(f"could not size a prompt within 1.5% of {target} tokens")


def counter(value):
    return value if type(value) is int and value >= 0 else None


def positive_number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def value_from(primary, backup):
    return primary if primary is not None else backup


def stream_completion(port, prompt, max_tokens, timeout, barrier=None):
    payload = {"model": MODEL, "messages": messages(prompt), "max_tokens": max_tokens,
               "temperature": 0, "chat_template_kwargs": TEMPLATE, "stream": True,
               "stream_options": {"include_usage": True}}
    if barrier:
        barrier.wait(timeout=timeout)
    start = time.monotonic()
    first, last = None, None
    pieces = []
    usage, engine = {}, {}
    finish_reason = None
    try:
        with timed_connection(port, timeout) as connection, request(
            connection, "/v1/chat/completions", payload, stream=True,
        ) as response:
            while True:
                if time.monotonic() - start > timeout:
                    raise ApiError(f"completion exceeded {timeout} seconds")
                line = response.readline(STREAM_LINE_LIMIT + 1)
                if len(line) > STREAM_LINE_LIMIT:
                    raise ApiError("stream event exceeded the size limit")
                if not line:
                    raise ApiError("completion stream closed before [DONE]")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if not isinstance(chunk, dict) or "error" in chunk:
                    raise ApiError(f"unexpected stream event: {data[:200]!r}")
                for choice in chunk.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str) and content:
                        last = time.monotonic()
                        if first is None:
                            first = last
                        pieces.append(content)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                if isinstance(chunk.get("usage"), dict):
                    usage.update(chunk["usage"])
                if isinstance(chunk.get("tensorfold"), dict):
                    engine.update(chunk["tensorfold"])
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise ApiError(f"stream at {HOST}:{port}: {error}") from error
    end = time.monotonic()
    if first is None or finish_reason not in {"stop", "length"}:
        raise ApiError("completion stream has no answer text or expected finish reason")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = {}
    cached = value_from(counter(details.get("cached_tokens")), counter(engine.get("cached")))
    prompt_tokens = value_from(counter(usage.get("prompt_tokens")), counter(engine.get("prompt_tokens")))
    output_tokens = value_from(counter(usage.get("completion_tokens")), counter(engine.get("completion_tokens")))
    if not prompt_tokens or not output_tokens:
        raise ApiError("stream did not report explicit positive prompt and completion token counts")
    prefill = value_from(positive_number(engine.get("prefill_seconds")),
                         positive_number(engine.get("prefill_s")))
    decode = value_from(positive_number(engine.get("tokens_per_second")),
                        positive_number(engine.get("decode_tps")))
    decode_seconds = value_from(positive_number(engine.get("decode_seconds")),
                                positive_number(engine.get("decode_s")))
    if decode is None and decode_seconds is not None:
        decode = output_tokens / decode_seconds
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": prompt_tokens, "completion_tokens": output_tokens,
        "cached_prompt_tokens": cached, "finish_reason": finish_reason,
        "elapsed_seconds": end - start, "first_content_seconds": first - start,
        "last_content_seconds": last - start, "client_completion_tokens_per_second": output_tokens / (end - start),
        "engine_prefill_seconds": prefill, "engine_decode_seconds": decode_seconds,
        "engine_decode_tokens_per_second": decode,
        "uncached_prompt_tokens_per_second": (prompt_tokens - cached) / prefill
        if prefill and cached is not None and 0 <= cached < prompt_tokens else None,
    }, "".join(pieces)


def decode_batch(port, seed, streams, timeout, workload="inventory"):
    barrier = threading.Barrier(streams)
    inputs = [output_prompt(seed, stream, workload) for stream in range(streams)]
    with ThreadPoolExecutor(max_workers=streams) as workers:
        begin = time.monotonic()
        futures = [workers.submit(stream_completion, port, text, COMPLETION_BUDGET, timeout, barrier)
                   for text in inputs]
        results = [future.result()[0] for future in futures]
        wall = time.monotonic() - begin
    return {"streams": streams, "workload": workload, "batch_elapsed_seconds": wall,
            "batch_client_completion_tokens_per_second": sum(row["completion_tokens"] for row in results) / wall,
            "median_first_content_seconds": statistics.median(row["first_content_seconds"] for row in results),
            "requests": results}


def benchmark(port, profile, seed, timeout):
    require_model(port)
    stream_counts, prompt_targets = PROFILES[profile]
    stream_completion(port, output_prompt(seed, -1), 16, timeout)
    decode_rows = []
    workloads = ("prose", "code") if profile == "full-decode" else ("inventory",)
    repeats = 3 if profile == "full-decode" else 1
    for workload_index, workload in enumerate(workloads):
        for streams in stream_counts:
            for repetition in range(repeats):
                print(f"{workload}: C{streams}, repeat {repetition + 1}/{repeats}",
                      file=sys.stderr, flush=True)
                trial_seed = seed + streams * 1009 + workload_index * 100000 + repetition * 10000
                batch = decode_batch(port, trial_seed, streams, timeout, workload)
                batch["repetition"] = repetition + 1
                decode_rows.append(batch)
    prefill_rows = []
    for target in prompt_targets:
        print(f"Running approximately {target:,} input tokens", file=sys.stderr, flush=True)
        prompt, expected, tokenized = size_prompt(port, seed + target, target, timeout)
        result, text = stream_completion(port, prompt, 32, timeout)
        if result["prompt_tokens"] != tokenized:
            raise ApiError(f"/tokenize counted {tokenized}, completion counted {result['prompt_tokens']}")
        recovered = text.strip() == expected
        if not recovered and profile != "full-prefill":
            raise ApiError(f"{tokenized}-token prompt returned an incorrect audit code")
        result.update(target_prompt_tokens=target, tokenized_prompt_tokens=tokenized,
                      audit_code_matched=recovered)
        prefill_rows.append(result)
    return {
        "schema_version": 1, "endpoint": f"http://{HOST}:{port}/v1", "model": MODEL,
        "profile": profile, "seed": seed, "request_timeout_seconds": timeout,
        "thinking": False, "temperature": 0, "decode_max_tokens": COMPLETION_BUDGET,
        "decode_repeats_per_case": repeats,
        "notes": [
            "Client elapsed time includes loopback HTTP, scheduling, prefill, decode, and final usage.",
            "First content is the first nonempty streamed text fragment.",
            "Batch client rate is all reported completion tokens divided by measured batch wall time.",
            "Engine counters are nullable. Prefill rate excludes reported cached prompt tokens.",
            "A repeated seed may hit prompt cache. Use a new seed for a fresh run.",
            "The full-decode profile repeats each workload and concurrency three times.",
        ],
        "completion_batches": decode_rows, "long_prompts": prefill_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=600, help="seconds per request, 1–1800")
    parser.add_argument("--output", type=Path, help="save JSON to a new file, otherwise print it")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800:
        parser.error("--timeout must be between 1 and 1800 seconds")
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        results = json.dumps(benchmark(args.port, args.profile, args.seed, args.timeout),
                             indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x") as destination:
                destination.write(results + "\n")
        else:
            print(results)
    except (ApiError, OSError, threading.BrokenBarrierError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
