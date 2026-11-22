# Qwen3.8 Flash Next on one DGX Spark

Serve NVIDIA’s **Qwen3.8 Flash Next NVFP4** checkpoint with **TensorFold** on one 128 GB DGX Spark. The tested profile keeps the API on localhost and supports chat, native images, and native video in a single model.

[How the model works](docs/how-it-works.md) · [Quick start](#quick-start) · [Reproduce](docs/reproduce.md) · [Troubleshooting](docs/troubleshooting.md) · [Results](results/README.md) · [Benchmarks](#benchmarks)

## Quick start

Run these commands on **one NVIDIA DGX Spark (GB10, Linux ARM64)**. You need Python 3.9+, Bash 4.4+, Git, curl, Docker Engine with local socket access, and the NVIDIA container runtime. For a fresh install, allow about **160 GiB free disk** and **107 GiB available unified memory** at startup. [Host checklist and storage setup](docs/reproduce.md#before-you-start)

```sh
git clone https://github.com/outsmartchad/qwen38-flash-next-single-dgx-spark-tensorfold.git &&
cd qwen38-flash-next-single-dgx-spark-tensorfold &&
python3 bin/recipe.py preflight &&
python3 bin/recipe.py fetch &&
python3 bin/recipe.py prepare &&
python3 bin/recipe.py start &&
python3 bin/recipe.py status &&
python3 bin/recipe.py smoke
```

The first `prepare` builds the pinned TensorFold image and downloads about 96 GiB of weights. If it is interrupted, rerun `prepare` with the same paths to reuse its caches. When `smoke` reports `Qwen3.8-Flash-Next returned SPARK_OK`, point your OpenAI-compatible client at **`http://127.0.0.1:18889/v1`** and use **`Qwen3.8-Flash-Next`** as the model ID. For a client on another machine, use the [SSH forwarding example](docs/reproduce.md#connect-an-api-client).

```sh
# Optional image and video check
python3 tools/multimodal_smoke.py

# Stop the recipe container when you finish, keeping the image and weights
python3 bin/recipe.py stop
```

On November 22, 2026, a fresh clone built the TensorFold image, staged the pinned weights in an empty cache, and answered chat on one DGX Spark. The benchmark tables and native media checks below come from earlier resident runs. [What the validation covered](results/README.md)

## Benchmarks

All tests ran on a resident Flash Next instance on one GB10 Spark using TensorFold 1.2.7 with MTP enabled. Each decode row shows the fastest aggregate run out of three at that concurrency. **Per request** and **TTFT** are medians across the streams in that same run.

### Decode, prose (thinking on)

Direct loopback requests used the SparkDash hash-map prompt, 400 output tokens per request, temperature 0.7, top-p 1, and top-k 20. **Aggregate** measures decode across the shared streaming window. **Per request** is the measured streaming decode rate. TTFT includes the first reasoning token. The one- and five-stream rows are the results used in the showcase card.

| Concurrent requests | Best aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 56.0 tok/s | 56.0 tok/s | 63 ms |
| 2 | 82.6 tok/s | 44.2 tok/s | 90 ms |
| 3 | 102.3 tok/s | 35.6 tok/s | 120 ms |
| 4 | 119.9 tok/s | 31.5 tok/s | 150 ms |
| 5 | 125.6 tok/s | 25.8 tok/s | 181 ms |

[Thinking-on prose data](results/decode-prose-thinking-on-2026-11-20.json) · [Methodology](docs/benchmarking.md#thinking-on-prose)

### Decode, prose (thinking off)

The thinking-off tables use fresh synthetic prompts from the published decode sweep, with 58–72 input tokens, 256 output tokens, and temperature zero. **Aggregate** is whole-batch wall throughput. **Per request** uses the engine decode counter. TTFT starts at the request and ends at the first streamed answer text.

| Concurrent requests | Best aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 50.1 tok/s | 51.6 tok/s | 166 ms |
| 2 | 71.9 tok/s | 37.3 tok/s | 252 ms |
| 3 | 90.8 tok/s | 32.2 tok/s | 337 ms |
| 4 | 108.2 tok/s | 28.5 tok/s | 412 ms |
| 5 | 115.6 tok/s | 24.7 tok/s | 509 ms |

### Decode, code (thinking off)

| Concurrent requests | Best aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 81.5 tok/s | 85.7 tok/s | 162 ms |
| 2 | 119.7 tok/s | 65.1 tok/s | 216 ms |
| 3 | 145.7 tok/s | 52.6 tok/s | 303 ms |
| 4 | 166.3 tok/s | 45.6 tok/s | 386 ms |
| 5 | 174.5 tok/s | 37.9 tok/s | 458 ms |

### Prefill, fresh prompts

| Prompt tokens | Prefill | TTFT |
| ---: | ---: | ---: |
| 4,084 | 2,292 tok/s | 1.79 s |
| 8,177 | 2,387 tok/s | 3.43 s |
| 16,366 | 2,351 tok/s | 6.97 s |
| 32,767 | 2,343 tok/s | 14.00 s |
| 65,544 | 2,283 tok/s | 28.73 s |
| 131,056 | 2,195 tok/s | 59.76 s |
| 200,007 | 2,099 tok/s | 95.34 s |

The prefill rate divides uncached input tokens by engine prefill time. Each prompt reused two template tokens. All seven runs returned the correct retrieval marker. Five-stream decode and the longest-input check were separate workloads.

[Decode data](results/decode-memory-2026-11-20.json) · [Earlier decode run](results/full-decode-2026-11-20.json) · [Prefill data](results/full-prefill-2026-11-20.json) · [How to reproduce](docs/reproduce.md#run-the-benchmarks) · [45 archived tuning runs](docs/historical-benchmarks.md)

The layout follows [Mia’s performance tables](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold#performance). Mia’s SparkDash headline figures were measured on the earlier TensorFold 1.1.4 build.

## Images and video

Flash Next has its own vision encoder. Send an image as an OpenAI-style `image_url` content part. This recipe enables native video too. It accepts an inline MP4 as a `video_url` part, decodes it with the bundled local decoder, and samples visual frames. [NVIDIA model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)

### Paired image spot check

With the same eight generated PNGs and questions on one Spark, Flash Next answered **7/8** exactly and Qwen3-VL 8B answered **8/8**. Both backends ran locally at temperature zero with a 320-token generation limit. The score uses the same exact-answer normalization for both. This small test covers a mix of OCR and visual reasoning tasks.

| Image task | Flash Next | Qwen3-VL 8B |
| --- | :---: | :---: |
| Clean equipment label | Pass | Pass |
| Faint, small serial | Pass | Pass |
| Rotated label | Miss | Pass |
| Traditional Chinese sign | Pass | Pass |
| Receipt total | Pass | Pass |
| Table row and column | Pass | Pass |
| Relative bar height | Pass | Pass |
| Object position | Pass | Pass |

Flash Next read small print, traditional Chinese, receipts, and tables. It also handled the simple chart and spatial question. Qwen3-VL recovered the rotated label that Flash Next missed. Rotating the same image upright let Flash Next read the serial correctly. This supports using Flash Next for routine image requests and keeping the second model available for difficult cases.

[Answers, prompts, and timings](results/vision-comparison-2026-11-20.json) · [Spark multimodal setup](https://github.com/outsmartchad/dgx-spark-setup)

### Native media smoke check

The included PNG and 12-second MP4 returned **red** and **red → blue → green** respectively. Run `python3 tools/multimodal_smoke.py` against a running server. [Media smoke data](results/native-media-smoke.json) · [Test fixtures](fixtures/README.md)

## Hardware and runtime

- **Tested host** — one NVIDIA GB10 DGX Spark with 128 GB unified memory. Linux reported 121.69 GiB total on our machine. The run record captures Ubuntu, kernel, NVIDIA driver, Docker, and runtime versions. [Observed configuration](results/running-profile.json) · [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- **Checkpoint** — NVIDIA’s NVFP4 quantization-aware distillation checkpoint, approximately 96 GiB on disk. The model and the engine’s FP8 KV cache use separate formats. [Pinned NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300)
- **Serving profile** — TensorFold v1.2.7 with Mia’s patches, five configured slots, 262,144 configured context tokens, FP8 KV cache, and MTP with up to six drafts per round. Native images and video are enabled. The server uses a 9 GiB memory reserve. Performance and capacity depend on available unified memory, prompt length, output budget, cache state, and concurrency. [Pinned upstream recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9)

Start with a quiet Spark and around **107 GiB of available memory** for this serving profile, as estimated by upstream. Allow roughly **160 GiB of free storage** for the checkpoint and build, with space on both the model-cache and Docker filesystems. The model’s boot-time memory must fit in unified memory alongside the OS and other processes. We observed 32.50 GiB of Linux `MemAvailable` after the server was up and idle, a single snapshot rather than a startup or peak measurement. [Upstream requirements](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/README.md#requirements)

## Validation

| Check | Result | Last reported summary |
| --- | --- | --- |
| Typed, streamed and continued tool calls | Pass | toolcheck: 3/3 passed |
| Prompt plus reply-budget admission at 262,144 | Pass | context_boundary: PASS: 16 checks; prompt=20, context=262144 |
| Four follow-ups on a roughly 30K conversation | Pass | prompt_reuse: PASS: every follow-up resumed >= 90% of its prompt (worst 99.3%) |
| Greedy parity with drafting disabled and concurrent requests | Pass | PASS conc: 5 concurrent replies equal the one-at-a-time ones (server max_batch_size 5) |

[Upstream check receipts](results/validation-2026-11-20.json) · [Native image and video checks](results/native-media-smoke.json)

## Build another recipe

The [single-Spark backlog](docs/model-backlog.md) records the candidate checkpoints, engines, sources, and hardware evidence. It includes the 27B comparator, alternate Flash Next serving paths, and larger-model candidates that need compression or offload. [Recipe template](docs/reusing-this-recipe.md)

If you want to wrap the server with Telegram, voice, mesh generation, and video generation, see the [DGX Spark multimodal setup](https://github.com/outsmartchad/dgx-spark-setup).

## Credits and license

Thanks to [Mia’s AI Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) for the upstream recipe and patches, [TensorFold contributors](https://github.com/ashhart/TensorFold) for the engine, [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) for the quantized checkpoint, and [the Qwen team](https://github.com/QwenLM) for the model family. [Sybil Solutions](https://github.com/sybil-solutions) informed the reproducibility and benchmark format.

Our scripts, documentation, and synthetic fixtures are [MIT licensed](LICENSE). The upstream build, NGC image, and model retain their own terms. The wrapper fetches those inputs from their publishers. [Upstream notices](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/NOTICE)
