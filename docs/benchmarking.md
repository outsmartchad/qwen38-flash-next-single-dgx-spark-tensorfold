# Benchmark methodology

Run these commands on the Spark against the loopback endpoint. The script sends synthetic messages, requests token usage on the stream, and stores counters and prompt hashes. It makes one small warm-up request before the measured workload.

```sh
python3 bin/benchmark.py --profile quick --seed 2026112001 --output quick.json
python3 bin/benchmark.py --profile standard --seed 2026112002 --output standard.json
```

The `quick` profile covers one and five concurrent requests and approximately 4K input tokens. `standard` covers one, three, and five requests plus approximately 4K, 32K, and 128K inputs. `long` adds approximately 200K. Concurrency runs generate up to 256 tokens per short request. The long inputs run one at a time and check that the returned marker matches exactly. The script uses the server’s tokenizer and confirms its count against completion usage.

All profiles request thinking disabled and temperature zero. MTP uses the server’s launch settings. For fresh-prompt runs, choose a new seed. Using the same seed can reuse a cached prefix. The script reports cached input tokens when available. Run with other inference clients idle and repeat each workload with new seeds before reporting aggregate statistics.

## Fields and interpretation

- **First content** measures from the client request start to the first nonempty streamed content chunk. It includes scheduling, prefill, and initial decode.
- **Client completion rate** divides total reported completion tokens by the full request elapsed time, including prefill and final usage.
- **Engine decode rate** uses the backend’s reported decode counters when available. It is the metric for decode-only comparisons.
- **Aggregate batch rate** divides the sum of completion tokens across streams by their shared wall time. The batch starts its request threads together.
- **Reported prompt tokens** include chat formatting. The script reports both requested prompt size and actual token count for retrieval tests.

Keep this recipe’s one-slot results, five-short-stream throughput, and single-request long-context results distinct. To establish a concurrent long-context capacity, run the relevant combined workload and retain admission errors, per-request token budgets, and shared wall throughput.

The published [running profile](../results/running-profile.json) identifies the hardware and startup settings. The [native media smoke](../results/native-media-smoke.json) measures the elapsed time of single image and video correctness checks. Media times include request processing, and timestamp accuracy depends on frame sampling.
