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
    "prose-thinking-on": ((1, 2, 3, 4, 5), ()),
    "full-prefill": ((), (4096, 8192, 16384, 32768, 65536, 131072, 200000)),
}
TEMPLATE = {"enable_thinking": False}
COMPLETION_BUDGET = 256
THINKING_PROSE_BUDGET = 400
# SparkDash's pinned DECODE_PROSE_PROMPT, linked in docs/benchmarking.md.
HASH_MAP_PROMPT = ("Write a detailed step-by-step explanation of how a hash map works, "
                   "including collision handling, resizing, and time complexity. Be thorough.")
MEMORY_SAMPLE_SECONDS = 0.05
MEMORY_IDLE_SAMPLES = 20
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


def stream_completion(port, prompt, max_tokens, timeout, barrier=None, *,
                      thinking=False, force_length=False):
    payload = {"model": MODEL, "messages": messages(prompt), "max_tokens": max_tokens,
               "temperature": 0, "chat_template_kwargs": TEMPLATE, "stream": True,
               "stream_options": {"include_usage": True}}
    if thinking:
        payload.update(temperature=0.7, top_p=1, top_k=20,
                       chat_template_kwargs={"enable_thinking": True})
    if force_length:
        payload.update(min_tokens=max_tokens, ignore_eos=True, stop=[])
    if barrier:
        barrier.wait(timeout=timeout)
    start = time.monotonic()
    first, last, first_answer = None, None, None
    reasoning_seen = False
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
                    delta = choice.get("delta") or {}
                    answer = delta.get("content")
                    reasoning = thinking and (delta.get("reasoning_content") or delta.get("reasoning"))
                    has_answer = isinstance(answer, str) and bool(answer)
                    has_reasoning = isinstance(reasoning, str) and bool(reasoning)
                    if has_answer or has_reasoning:
                        last = time.monotonic()
                        if first is None:
                            first = last
                        reasoning_seen |= has_reasoning
                    if has_answer:
                        if first_answer is None:
                            first_answer = last
                        pieces.append(answer)
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
        raise ApiError("completion stream has no text or expected finish reason")
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
    result = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": prompt_tokens, "completion_tokens": output_tokens,
        "cached_prompt_tokens": cached, "finish_reason": finish_reason,
        "elapsed_seconds": end - start, "first_content_seconds": first - start,
        "last_content_seconds": last - start, "client_completion_tokens_per_second": output_tokens / (end - start),
        "engine_prefill_seconds": prefill, "engine_decode_seconds": decode_seconds,
        "engine_decode_tokens_per_second": decode,
        "uncached_prompt_tokens_per_second": (prompt_tokens - cached) / prefill
        if prefill and cached is not None and 0 <= cached < prompt_tokens else None,
    }
    if thinking:
        result.update(first_event_time=first, last_event_time=last,
                      first_answer_seconds=None if first_answer is None else first_answer - start,
                      reasoning_seen=reasoning_seen)
    if force_length and output_tokens != max_tokens:
        raise ApiError(f"expected {max_tokens} output tokens, server reported {output_tokens}")
    return result, "".join(pieces)


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


def prose_wave(port, streams, repeat, timeout):
    inputs = [HASH_MAP_PROMPT] if streams == 1 else [
        f"Request ID: decode-{index}\n{HASH_MAP_PROMPT}" for index in range(1, streams + 1)
    ]
    barrier = threading.Barrier(streams)
    with ThreadPoolExecutor(max_workers=streams) as workers:
        runs = [workers.submit(stream_completion, port, prompt, THINKING_PROSE_BUDGET,
                               timeout, barrier, thinking=True, force_length=True) for prompt in inputs]
        results = [run.result()[0] for run in runs]
    first_events, last_events = [], []
    requests = []
    for result in results:
        first = result["first_event_time"]
        last = result["last_event_time"]
        if not result["reasoning_seen"] or last <= first:
            raise ApiError("stream did not contain timed reasoning and decode events")
        first_events.append(first)
        last_events.append(last)
        requests.append({
            "completion_tokens": result["completion_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "prompt_sha256": result["prompt_sha256"],
            "cached_tokens": result["cached_prompt_tokens"],
            "stream_decode_tok_s": (result["completion_tokens"] - 1) / (last - first),
            "ttft_any_ms": result["first_content_seconds"] * 1000,
            "ttft_answer_ms": None if result["first_answer_seconds"] is None
            else result["first_answer_seconds"] * 1000,
        })
    decode_seconds = max(last_events) - min(first_events)
    generated_after_first = sum(row["completion_tokens"] - 1 for row in requests)
    return {"streams": streams, "repeat": repeat, "requests": requests,
            "aggregate_decode_tok_s": generated_after_first / decode_seconds}


def thinking_prose_benchmark(port, stream_counts, timeout):
    stream_completion(port, HASH_MAP_PROMPT, 32, timeout, thinking=True)
    batches = []
    best_by_concurrency = []
    for streams in stream_counts:
        waves = []
        for repeat in range(1, 4):
            print(f"Thinking-on prose, C{streams}, repeat {repeat}/3", file=sys.stderr, flush=True)
            waves.append(prose_wave(port, streams, repeat, timeout))
        batches.extend(waves)
        best = max(waves, key=lambda wave: wave["aggregate_decode_tok_s"])
        best_by_concurrency.append({
            "streams": streams, "repeat": best["repeat"],
            "aggregate_decode_tok_s": best["aggregate_decode_tok_s"],
            "median_stream_decode_tok_s": statistics.median(
                row["stream_decode_tok_s"] for row in best["requests"]
            ),
            "median_ttft_any_ms": statistics.median(row["ttft_any_ms"] for row in best["requests"]),
        })
    return {
        "schema_version": 1, "profile": "prose-thinking-on",
        "endpoint": f"http://{HOST}:{port}/v1/chat/completions", "model": MODEL,
        "workload": "SparkDash hash-map prose", "thinking": True, "temperature": 0.7,
        "prompt_source": ("https://github.com/MiaAI-Lab/sparkDash/blob/"
                          "91a3e46d7cbadf6daf4d53b5e7ae34e519f353ca/src/shared/llmPrompts.js#L53-L56"),
        "top_p": 1, "top_k": 20, "completion_tokens_per_request": THINKING_PROSE_BUDGET,
        "repeats_per_concurrency": 3, "batches": batches, "best_by_concurrency": best_by_concurrency,
    }


def host_memory():
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {field: int(meminfo[field].split()[0]) for field in ("MemTotal", "MemAvailable", "SwapFree")}


def decode_batch_with_memory(port, seed, streams, timeout, workload):
    idle = []
    for _ in range(MEMORY_IDLE_SAMPLES):
        idle.append(host_memory())
        time.sleep(MEMORY_SAMPLE_SECONDS)
    active = []
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(decode_batch, port, seed, streams, timeout, workload)
        while not future.done():
            active.append(host_memory())
            time.sleep(MEMORY_SAMPLE_SECONDS)
        result = future.result()
    if not active:
        raise ApiError("decode finished before any host memory samples were captured")
    available_idle = [sample["MemAvailable"] for sample in idle]
    available_active = [sample["MemAvailable"] for sample in active]
    result["host_memory"] = {
        "linux_memtotal_kib": idle[0]["MemTotal"],
        "idle_samples": len(idle),
        "idle_memavailable_median_kib": statistics.median(available_idle),
        "idle_memavailable_min_kib": min(available_idle),
        "idle_memavailable_max_kib": max(available_idle),
        "batch_samples": len(active),
        "batch_memavailable_min_kib": min(available_active),
        "idle_swapfree_kib": idle[-1]["SwapFree"],
        "batch_swapfree_min_kib": min(sample["SwapFree"] for sample in active),
    }
    return result


def benchmark(port, profile, seed, timeout, sample_memory=False):
    require_model(port)
    stream_counts, prompt_targets = PROFILES[profile]
    if profile == "prose-thinking-on":
        return thinking_prose_benchmark(port, stream_counts, timeout)
    stream_completion(port, output_prompt(seed, -1), 16, timeout)
    decode_rows = []
    run_decode = decode_batch_with_memory if sample_memory else decode_batch
    workloads = ("prose", "code") if profile == "full-decode" else ("inventory",)
    repeats = 3 if profile == "full-decode" else 1
    for workload_index, workload in enumerate(workloads):
        for streams in stream_counts:
            for repetition in range(repeats):
                print(f"{workload}: C{streams}, repeat {repetition + 1}/{repeats}",
                      file=sys.stderr, flush=True)
                trial_seed = seed + streams * 1009 + workload_index * 100000 + repetition * 10000
                batch = run_decode(port, trial_seed, streams, timeout, workload)
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
    results = {
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
    if sample_memory:
        results["host_memory_sampling"] = {
            "source": "/proc/meminfo on the server host",
            "target_sampling_interval_seconds": MEMORY_SAMPLE_SECONDS,
            "idle_samples_per_batch": MEMORY_IDLE_SAMPLES,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=600, help="seconds per request, 1–1800")
    parser.add_argument("--sample-memory", action="store_true",
                        help="sample Linux host RAM during each decode batch")
    parser.add_argument("--output", type=Path, help="save JSON to a new file, otherwise print it")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800:
        parser.error("--timeout must be between 1 and 1800 seconds")
    if args.sample_memory and (not PROFILES[args.profile][0] or args.profile == "prose-thinking-on"):
        parser.error("--sample-memory requires a thinking-off decode workload")
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        measurements = benchmark(args.port, args.profile, args.seed, args.timeout, args.sample_memory)
        results = json.dumps(measurements, indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x") as destination:
                destination.write(results + "\n")
        else:
            print(results)
    except (ApiError, OSError, threading.BrokenBarrierError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
