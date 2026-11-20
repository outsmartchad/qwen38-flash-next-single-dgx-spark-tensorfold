# Benchmark methodology

Run these commands on the Spark against the loopback endpoint. The script sends synthetic messages, requests token usage on the stream, and stores counters and prompt hashes. It makes one small warm-up request before the measured workload.

```sh
python3 bin/benchmark.py --profile quick --seed 2026112001 --output quick.json
python3 bin/benchmark.py --profile standard --seed 2026112002 --output standard.json

# The publication matrix, run sequentially on the same resident server.
python3 bin/benchmark.py --profile full-decode --seed 2026112031 --output full-decode.json
python3 bin/benchmark.py --profile full-prefill --seed 2026112032 --output full-prefill.json

# Repeat the decode matrix while recording RAM on the Spark itself.
python3 bin/benchmark.py --profile full-decode --seed 2026112041 --sample-memory --output decode-memory.json
```

The `quick` profile covers one and five concurrent requests and approximately 4K input tokens. `standard` covers one, three, and five requests plus approximately 4K, 32K, and 128K inputs. `long` adds approximately 200K. Concurrency runs generate up to 256 tokens per short request. The long inputs run one at a time and check that the returned marker matches exactly. The script uses the server’s tokenizer and confirms its count against completion usage.

`full-decode` runs separate prose tutorials and Python implementations at C1–C5, with three seeded repetitions per point and different prompts for simultaneous requests. `full-prefill` measures 4K, 8K, 16K, 32K, 64K, 128K, and approximately 200,000 input tokens once each. It reports each retrieval outcome with the timing so a miss stays visible. Run the two profiles sequentially to avoid workload contention.

All profiles request thinking disabled and temperature zero. MTP uses the server’s launch settings. For fresh-prompt runs, choose a new seed. Using the same seed can reuse a cached prefix. The script reports cached input tokens when available. Run with other inference clients idle and repeat each workload with new seeds before reporting aggregate statistics.

## Fields and interpretation

- **First content** measures from the client request start to the first nonempty streamed content chunk. It includes scheduling, prefill, and initial decode.
- **Client completion rate** divides total reported completion tokens by the full request elapsed time, including prefill and final usage.
- **Engine decode rate** uses the backend’s reported decode counters when available. It is the metric for decode-only comparisons.
- **Aggregate batch rate** divides the sum of completion tokens across streams by their shared wall time. The batch starts its request threads together.
- **Reported prompt tokens** include chat formatting. The script reports both requested prompt size and actual token count for retrieval tests.

Keep this recipe’s one-slot results, five-short-stream throughput, and single-request long-context results distinct. To establish a concurrent long-context capacity, run the relevant combined workload and retain admission errors, per-request token budgets, and shared wall throughput.

The published [running profile](../results/running-profile.json) identifies the hardware and startup settings. The [native media smoke](../results/native-media-smoke.json) measures the elapsed time of single image and video correctness checks. Media times include request processing, and timestamp accuracy depends on frame sampling.

## RAM under decode load

With `--sample-memory`, the benchmark samples Linux `/proc/meminfo` on the Spark about every 50 ms. Each batch uses 20 samples over the preceding second for its idle baseline, with Flash Next already loaded. **RAM left** is the median across repeats of each batch’s minimum `MemAvailable`. **Extra RAM** is the median drop from each batch’s own idle median to its minimum while running. **Δ vs prior** subtracts the extra RAM in the preceding concurrency row, with loaded idle as the baseline for one request.

`MemAvailable` estimates system-wide headroom on the unified-memory host. It includes the effects of Linux reclaimable caches and concurrent system activity. Small or negative steps can reflect variation between samples. The data also records `SwapFree` to flag changes. Run without other inference clients or media generation, and assess long contexts and image/video workloads separately.

The running TensorFold log reported **4.83 GiB per configured stream slot** with a **262,144-token window**, FP8 main KV and MTP enabled. The five configured windows account for about **24.2 GiB of the engine’s startup KV budget**. Adding a configured slot and sending an extra short request to an existing slot are different measurements. [Pinned upstream KV and memory notes](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/README.md#kv-and-memory)

## Upstream API checks

After the performance profiles, run the same targeted upstream scripts from the pinned Mia checkout. The commands below use a Spark shell and its loopback port. They send synthetic API requests.

```sh
python3 bin/recipe.py fetch
upstream="$(python3 bin/recipe.py plan | python3 -c 'import json,sys; print(json.load(sys.stdin)["source_dir"])')"
export API_URL=http://127.0.0.1:18889
export MODEL=Qwen3.8-Flash-Next
python3 "$upstream/tools/toolcheck.py"
python3 "$upstream/tools/context_boundary.py" --context 262144
python3 "$upstream/tools/prompt_reuse.py" 30000 4
python3 "$upstream/tools/exact.py" --prompts 5 --max-tokens 96 --pad 3000
```

The context check validates admission at the configured prompt-plus-reply budget with a short reply. The separate long-prefill profile exercises actual long input. Greedy parity compares observed responses with drafting enabled, disabled, and under contention.

## Raw runs

- [Full prose and code decode with RAM sampling](../results/decode-memory-2026-11-20.json)
- [Earlier full decode run](../results/full-decode-2026-11-20.json)
- [Full prefill and exact retrieval](../results/full-prefill-2026-11-20.json)
- [Pinned upstream validation](../results/validation-2026-11-20.json)
- [Quick inventory](../results/quick-2026-11-20.json) and [standard inventory](../results/standard-2026-11-20.json)
- [Historical benchmark tables](historical-benchmarks.md) and [45 archived runs in JSON](../results/historical-2026-11-19.json)

The presentation follows [Mia’s performance notes](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold#performance). Our workload uses defined prose and Python prompts at temperature zero with thinking off. Mia’s sparkDash prose runs enable thinking and use a different sampler. Compare engines only with the same prompts, sampling settings, and throughput definition.
