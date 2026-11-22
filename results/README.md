# Results and raw data

These measurements were collected on one NVIDIA DGX Spark with TensorFold 1.2.7 and the pinned NVIDIA Qwen3.8 Flash Next checkpoint. [Hardware and runtime](../README.md#hardware-and-runtime) · [Run the same workloads](../docs/reproduce.md#run-the-benchmarks)

| Workload | Published result and raw data |
| --- | --- |
| Thinking-on prose decode, 1–5 streams | **56.0 tok/s** at 1 stream and **125.6 tok/s aggregate** at 5 streams. [Raw data](decode-prose-thinking-on-2026-11-20.json) · [Method](../docs/benchmarking.md#thinking-on-prose) |
| Thinking-off prose and code decode, 1–5 streams | Best batch wall throughput and per-stream rates, including memory sampling. [Raw data](decode-memory-2026-11-20.json) · [Earlier sweep](full-decode-2026-11-20.json) |
| Fresh prefill and exact retrieval | Approx. 4K to 200K input tokens, with retrieval outcomes and cache counters. [Raw data](full-prefill-2026-11-20.json) |
| Quick and standard inventory | Short-request checks and long-prompt marker recovery. [Quick](quick-2026-11-20.json) · [Standard](standard-2026-11-20.json) |
| Native media and paired image checks | Known colors in a PNG and a video, plus eight image tasks against Qwen3-VL 8B. [Media smoke](native-media-smoke.json) · [Paired image results](vision-comparison-2026-11-20.json) |
| Engine behavior | Tool calls, context admission, prompt reuse, and greedy parity. [Raw checks](validation-2026-11-20.json) |
| Historical tuning | 45 exploratory runs across engine and serving variants. [Tables](../docs/historical-benchmarks.md) · [Raw archive](historical-2026-11-19.json) |

The performance and media results come from the resident deployment described in the [running profile](running-profile.json).

On November 22, 2026, a fresh checkout built the pinned image and downloaded and verified all 21 weight shards in a dedicated empty cache on one DGX Spark. After the Docker preflight fix, the recipe started the model on `127.0.0.1:18889`. The text smoke returned `SPARK_OK`, the documented unified-memory prompt returned a complete answer, and `stop` removed the test container.

The decode tables use different timing windows across thinking-on and thinking-off modes. Compare the [methodology](../docs/benchmarking.md) before interpreting a throughput change.
