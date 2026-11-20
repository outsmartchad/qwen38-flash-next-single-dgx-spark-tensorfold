# Historical Flash Next benchmarks

The November 19 experiments ran on one NVIDIA GB10 DGX Spark over loopback. The [downloadable telemetry](../results/historical-2026-11-19.json) contains the recorded per-request timing, tokens, and available memory observations for all 45 short, matched-16K and tuning runs. The [current recipe benchmarks](../README.md#benchmarks) use a different measurement workload.

## TensorFold short-prompt and MTP sweep

TensorFold 1.2.7, NVIDIA QAD NVFP4 weights, FP8 main KV, one timed stream, five configured slots, and 262,144 configured context. Each start ran two repetitions of Python, explanatory prose, and structured JSON with greedy sampling, thinking disabled, and 256 generated tokens. Decode and TTFT are medians. Each arm also passed the three synthetic correctness checks. The control spread matters here. The cap-10 result is only 0.10 tok/s above the best control, much smaller than the spread across controls.

| Run | C1 decode tok/s | Code | Prose | Structured | TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control-before | 59.41 | 59.41 | 40.03 | 106.00 | 131 |
| mtp3 | 54.69 | 54.69 | 41.24 | 65.46 | 131 |
| control-middle | 50.29 | 39.81 | 41.21 | 106.12 | 133 |
| mtp4 | 57.72 | 57.72 | 39.65 | 76.50 | 132 |
| control-after | 59.44 | 59.44 | 27.18 | 105.53 | 133 |
| mtp10 | 60.22 | 60.22 | 41.45 | 108.94 | 130 |
| control-final | 60.12 | 60.12 | 41.39 | 106.52 | 130 |
| serial-no-mtp | 27.17 | 27.09 | 27.18 | 27.10 | 56 |

The first and last controls measured **59.41** and **60.12 tok/s**. On the first control, the 7,549-token retrieval prompt took **3.294 s** to first token on the first request and **0.071 s** on repeat.

## Matched 16K drafting and cache sweep

Each row is one fresh server process and six completions over two 16,380-token prompts. The first request for each prompt reused two template tokens. Repeats reused 16,373 prompt tokens. Temperature was zero with thinking disabled. All 78 completions produced 256 visible tokens, with matching greedy outputs for each prompt across settings. The historical controls used greedy draft cap 8 and threshold 0.4. The no-draft arm switched off drafting per request while the MTP weights stayed resident.

| Run | All decode tok/s | Fresh decode tok/s | Fresh TTFT s | Cached decode tok/s | Cached TTFT s | Minimum available RAM GiB | Engine peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control-a | 53.83 | 49.43 | 7.136 | 55.10 | 0.079 | 37.89 | 77.14 |
| control-b | 54.43 | 49.66 | 7.146 | 54.99 | 0.082 | 37.96 | 77.14 |
| control-c | 54.56 | 49.82 | 7.201 | 54.93 | 0.079 | 37.90 | 77.14 |
| control-d | 54.32 | 48.96 | 7.046 | 54.69 | 0.082 | 38.13 | 77.14 |
| control-e | 54.11 | 49.84 | 7.094 | 54.38 | 0.081 | 37.92 | 77.14 |
| control-f | 54.22 | 49.54 | 7.151 | 54.66 | 0.083 | 38.06 | 77.14 |
| control-final | 53.74 | 49.31 | 7.256 | 54.51 | 0.083 | 37.82 | 77.14 |
| depth4 | 54.76 | 49.98 | 7.148 | 55.60 | 0.080 | 38.19 | 77.14 |
| repeat-depth4 | 54.45 | 49.73 | 7.155 | 55.24 | 0.082 | 37.99 | 77.14 |
| depth10 | 54.44 | 49.66 | 7.169 | 54.68 | 0.082 | 38.10 | 77.14 |
| confidence20 | 52.49 | 48.97 | 7.398 | 53.82 | 0.082 | 38.12 | 77.14 |
| confidence65 | 52.81 | 47.99 | 7.203 | 53.87 | 0.082 | 38.08 | 77.14 |
| draft-off | 26.76 | 23.51 | 6.905 | 26.85 | 0.073 | 38.46 | 76.70 |

Pooling the seven controls gave **49.49 tok/s** on fresh 16K and **54.66 tok/s** on cached 16K. The draft-off arm returned **23.51** and **26.85 tok/s** respectively. The best cap-4 median did not repeat outside the control range.

## Prompt chunking and draft-head sweep

The baseline used 3,072 prompt rows per chunk, the affine4 draft head, BF16 draft-head KV, and FP8 main KV. Each row used six short and six matched 16K completions. The tested changes maintained byte-for-byte greedy output parity on matching prompts. Startup is the observed local restart duration of each experiment. System RAM and engine peak memory use separate counters.

| Run | Boot s | Short decode tok/s | Fresh 16K decode tok/s | Cached 16K decode tok/s | Fresh TTFT s | Engine peak GiB | Min available RAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk-1536 | 40.2 | 58.45 | 49.19 | 54.82 | 7.92 | 76.73 | 38.39 |
| chunk-4096 | 42.3 | 58.34 | 49.66 | 54.88 | 6.84 | 78.05 | 37.05 |
| chunk-6144 | 40.3 | 58.06 | 49.35 | 54.55 | 6.78 | 79.14 | 36.17 |
| chunk-8192 | 42.3 | 57.78 | 49.15 | 54.24 | 6.74 | 80.09 | 35.14 |
| control-a | 40.3 | 58.47 | 49.49 | 54.68 | 7.31 | 77.50 | 37.95 |
| control-after-chunk-4096 | 42.3 | 58.41 | 49.29 | 54.56 | 7.06 | 77.50 | 37.85 |
| control-after-chunk-6144 | 42.3 | 58.23 | 49.46 | 54.43 | 7.01 | 77.50 | 37.62 |
| control-after-chunk-8192 | 42.3 | 58.26 | 49.62 | 54.53 | 7.04 | 77.50 | 37.64 |
| control-b | 40.3 | 58.30 | 49.41 | 54.73 | 7.15 | 77.50 | 37.94 |
| control-c | 40.3 | 58.36 | 49.31 | 54.77 | 7.11 | 77.50 | 37.87 |
| control-d | 42.3 | 58.27 | 48.96 | 54.55 | 7.21 | 77.50 | 37.87 |
| control-e | 40.2 | 58.58 | 49.59 | 54.52 | 7.21 | 77.50 | 37.92 |
| large-control-a | 42.3 | 58.51 | 48.56 | 54.47 | 7.19 | 77.50 | 37.90 |
| large-control-b | 40.2 | 58.05 | 49.17 | 54.31 | 7.00 | 77.50 | 37.72 |
| large-control-c | 40.3 | 58.05 | 49.42 | 54.46 | 7.02 | 77.50 | 37.73 |
| mtp-head-bf16 | 40.2 | 50.53 | 45.58 | 49.83 | 7.41 | 77.76 | 37.71 |
| mtp-kv-fp8 | 40.3 | 53.41 | 45.59 | 51.01 | 7.17 | 77.45 | 37.69 |
| repeat-chunk-4096 | 40.3 | 58.06 | 48.90 | 54.72 | 6.91 | 78.05 | 37.00 |
| repeat-chunk-6144 | 40.3 | 58.14 | 49.22 | 54.53 | 7.20 | 79.14 | 36.16 |
| repeat-chunk-8192 | 42.3 | 58.45 | 49.49 | 54.05 | 6.81 | 80.09 | 34.58 |

Larger chunks shortened prefill on some runs. The 8,192-row repeat missed the agreed 3% improvement threshold against the faster adjacent baseline and used about 2.6 GiB more peak allocator memory. FP8 draft-head KV and BF16 draft-head weights slowed the tested workload. The serving profile kept the 3,072-row setting.

## Separate GGUF and TurboQuant experiment

The alternate engine was [llama-tq at fed4890](https://github.com/LL4nc33/llama-tq/tree/fed4890a4ccd775d5c04b921720eda947d990f7d) with [Unsloth UD-IQ3_XXS at f5d175d](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/f5d175de2cf7f720d45d6064019a81fcf056e986). A tensor type ID collided with the fork’s cache extension. We repacked three Q2_0 expert tensors losslessly into Q4_0 in a separate copy and used the same converted shards in each arm. The server used 32,768 context, one slot, 1,024-token batches, 256-token microbatches, Flash Attention, and no speculative draft. Each arm used six short and two fresh 16,380-token prompts at 256 visible output tokens.

| KV configuration | Short decode tok/s | 16K decode tok/s | 16K TTFT s | Main GPU KV MiB | Checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0, first control | 32.75 | 30.82 | 28.27 | 408 | 3/3 |
| KTQ4/VTQ4, default staging | 32.58 | 28.98 | 28.51 | 990 | 3/3 |
| Q8_0, repeat control | 33.60 | 31.19 | 28.22 | 408 | 3/3 |
| KTQ4/VTQ4, staging disabled | 31.87 | 30.06 | 28.74 | 222 | 3/3 |

The two Q8 controls matched all six outputs. Each TurboQuant arm matched Q8 on two of six and passed the three correctness checks. Main KV memory excludes the 96 MiB attention index cache common to all four arms. TurboQuant with staging disabled saved 186 MiB of main KV versus Q8_0. The NVIDIA-checkpoint TensorFold and vLLM configurations rejected the tested TurboQuant KV types, so those attempts have no throughput result.

## Earlier 27B comparison and hardware reference

One warm-server, single-stream comparison used ten measured requests, temperature zero, thinking enabled, and a 256-token generation budget. Flash Next's run used its completion budget for reasoning. These medians describe generation cost on the fixed workload.

| Model | Client decode tok/s | Engine decode tok/s | End-to-end tok/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 Flash Next | 50.97 | 50.96 | 49.81 | 135.8 |
| Qwen3.8 27B | 55.74 | 55.96 | 54.62 | 110.8 |

On the same device, standalone microbenchmarks measured **211.6 GB/s** median GPU copy bandwidth and **82.6 TFLOP/s** median BF16 GEMM throughput. These figures describe separate hardware workloads.
