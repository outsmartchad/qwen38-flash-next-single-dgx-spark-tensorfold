# Qwen3.8 Flash Next on one DGX Spark

Run NVIDIA’s quantized **Qwen3.8 Flash Next** on one 128 GB DGX Spark using **TensorFold**. This recipe pins the upstream build and checkpoint, serves an OpenAI-compatible API on localhost, and includes text, image, and native-video checks plus a streaming benchmark. The configuration is based on a working single-Spark deployment tested on 20 November 2026.

[Hardware and runtime](#hardware-and-runtime) · [Quick start](#quick-start) · [Benchmarks](#benchmarks) · [Model backlog](docs/model-backlog.md) · [Recipe template](docs/reusing-this-recipe.md)

## Hardware and runtime

- **Tested host** — one NVIDIA GB10 DGX Spark with 128 GB unified memory. Linux reported 121.69 GiB total on our machine. The run record captures Ubuntu, kernel, NVIDIA driver, Docker, and runtime versions. [Observed configuration](results/running-profile.json) · [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- **Checkpoint** — NVIDIA’s NVFP4 quantization-aware distillation checkpoint, approximately 96 GiB on disk. The model and the engine’s FP8 KV cache use separate formats. [Pinned NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300)
- **Serving profile** — TensorFold v1.2.7 with Mia’s patches, five configured slots, 262,144 configured context tokens, FP8 KV cache, and MTP with up to six drafts per round. Native images and video are enabled. The server uses a 9 GiB memory reserve. Performance and capacity depend on available unified memory, prompt length, output budget, cache state, and concurrency. [Pinned upstream recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9)

Start with a quiet Spark and around **107 GiB of available memory** for this serving profile, as estimated by upstream. Allow roughly **160 GB of free storage** for the checkpoint and build, with space on both the model-cache and Docker filesystems. The model’s boot-time memory must fit in unified memory alongside the OS and other processes. We observed 32.50 GiB of Linux `MemAvailable` after the server was up and idle, a single snapshot rather than a startup or peak measurement. [Upstream requirements](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/README.md#requirements)

## Quick start

Run on the Spark with Linux ARM64, Python 3.9+, Bash 4.4+, Git, curl, Docker Engine and a working NVIDIA container runtime. The wrapper also checks the host utilities it needs. Use SSH to run these commands from another machine.

```sh
git clone https://github.com/outsmartchad/qwen38-flash-next-single-dgx-spark-tensorfold.git
cd qwen38-flash-next-single-dgx-spark-tensorfold

python3 bin/recipe.py plan       # show effective inputs and settings
python3 bin/recipe.py preflight  # check the host
python3 bin/recipe.py fetch      # check out the pinned Mia source
python3 bin/recipe.py prepare    # build the image and stage the checkpoint
python3 bin/recipe.py start
python3 bin/recipe.py status
python3 bin/recipe.py smoke
python3 tools/multimodal_smoke.py
```

The first build downloads the base image and build inputs and stages the checkpoint. Use the same `--data-dir`, `--hf-cache`, and `--port` options for the lifecycle commands if you override them. `--hf-cache` takes the root of a dedicated Hugging Face cache for public weights. The API defaults to `http://127.0.0.1:18889/v1` and serves model ID `Qwen3.8-Flash-Next`. If another process owns the port, the wrapper refuses the new launch. Avoid loading a second copy just to run the tests. The status and smoke commands can check a compatible server that is already running.

Example request from a Spark shell:

```sh
curl -fsS --max-time 30 http://127.0.0.1:18889/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Explain unified memory in two sentences."}],"max_tokens":160,"reasoning_effort":"none"}'
```

To stop an instance launched by this recipe, use `python3 bin/recipe.py stop` with the same port and data directory.

## What this repository adds

This project starts from Mia’s single-Spark engine build. The wrapper fetches a pinned checkout and uses the upstream scripts to apply the five patches and stage the NVIDIA weights. The base NGC image is pinned by digest and the recipe builds under its own image tag and launches using the verified local image ID. Our additions are the loopback serving profile, lifecycle checks, multimodal regression fixtures, benchmark harness, and a record of the configuration we observed on hardware.

The [native media check](tools/multimodal_smoke.py) submits the bundled PNG and 12-second MP4 as `image_url` and `video_url` content parts. It checks that Flash Next sees the red square and reports the three colors in time order. The recorded run returned **red**, then **red → blue → green**, in 0.35 and 1.01 seconds respectively. This exercises the local FFmpeg-backed video path and checks visual order. [Native media results](results/native-media-smoke.json) · [Synthetic fixtures](fixtures/README.md)

## Benchmarks

Measured on the resident TensorFold 1.2.7 server on one GB10 Spark on November 20, 2026. The [run configuration](results/running-profile.json) records the image, checkpoint, memory reserve, native video settings, and five configured slots.

### Decode

Three runs at every stream count for each workload. All 90 requests generated 256 tokens, with temperature zero, thinking off, and MTP enabled. **Total** is median whole-batch wall throughput, including prefill and request overhead. **Per request** is median engine decode. First content is the median across batch medians.

| Streams | Prose total tok/s | Prose per request tok/s | Prose first content | Code total tok/s | Code per request tok/s | Code first content |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.9 | 47.0 | 156 ms | 81.0 | 84.4 | 140 ms |
| 2 | 67.7 | 35.9 | 241 ms | 111.9 | 61.3 | 222 ms |
| 3 | 86.6 | 30.9 | 328 ms | 137.4 | 50.0 | 301 ms |
| 4 | 103.4 | 27.1 | 413 ms | 160.1 | 44.2 | 379 ms |
| 5 | 106.7 | 23.3 | 492 ms | 164.2 | 36.6 | 458 ms |

[Full decode JSON](results/full-decode-2026-11-20.json)

### Fresh prefill and retrieval

One request at each input length. Rates divide uncached input tokens by the engine’s prefill time. First content is measured from the beginning of the HTTP request to the first streamed text.

| Input tokens | Cached tokens | Engine prefill tok/s | First content | Exact retrieval |
| ---: | ---: | ---: | ---: | :---: |
| 4,084 | 2 | 2,292 | 1.79 s | Pass |
| 8,177 | 2 | 2,387 | 3.43 s | Pass |
| 16,366 | 2 | 2,351 | 6.97 s | Pass |
| 32,767 | 2 | 2,343 | 14.00 s | Pass |
| 65,544 | 2 | 2,283 | 28.73 s | Pass |
| 131,056 | 2 | 2,195 | 59.76 s | Pass |
| 200,007 | 2 | 2,099 | 95.34 s | Pass |

[Full prefill JSON](results/full-prefill-2026-11-20.json)

### Validation

| Check | Result | Last reported summary |
| --- | --- | --- |
| Typed, streamed and continued tool calls | Pass | toolcheck: 3/3 passed |
| Prompt plus reply-budget admission at 262,144 | Pass | context_boundary: PASS: 16 checks; prompt=20, context=262144 |
| Four follow-ups on a roughly 30K conversation | Pass | prompt_reuse: PASS: every follow-up resumed >= 90% of its prompt (worst 99.3%) |
| Greedy parity with drafting disabled and concurrent requests | Pass | PASS conc: 5 concurrent replies equal the one-at-a-time ones (server max_batch_size 5) |

[Upstream check receipts](results/validation-2026-11-20.json) · [Native image and video checks](results/native-media-smoke.json)

Reproduce the matrix from a Spark shell:

```sh
python3 bin/benchmark.py --profile full-decode --seed 2026112031 --output full-decode.json
python3 bin/benchmark.py --profile full-prefill --seed 2026112032 --output full-prefill.json
```

[Methodology and earlier inventory runs](docs/benchmarking.md) · [All 45 archived runs, including MTP, chunking, and GGUF/TurboQuant](docs/historical-benchmarks.md)

The longest prompt and the five-stream decode tests ran as separate workloads. The context-boundary check tests prompt-plus-output admission using a short reply. A clean-machine build and independent reproduction are still open.

## Build another recipe

The [single-Spark backlog](docs/model-backlog.md) records the candidate checkpoints, engines, sources, and hardware evidence. It includes the 27B comparator, alternate Flash Next serving paths, and larger-model candidates that need compression or offload. [Recipe template](docs/reusing-this-recipe.md)

If you want to wrap the server with Telegram, voice, mesh generation, and video generation, see the [DGX Spark multimodal setup](https://github.com/outsmartchad/dgx-spark-setup).

## Credits and license

Thanks to [Mia’s AI Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) for the upstream recipe and patches, [TensorFold contributors](https://github.com/ashhart/TensorFold) for the engine, [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) for the quantized checkpoint, and [the Qwen team](https://github.com/QwenLM) for the model family. [Sybil Solutions](https://github.com/sybil-solutions) informed the reproducibility and benchmark format.

Our scripts, documentation, and synthetic fixtures are [MIT licensed](LICENSE). The upstream build, NGC image, and model retain their own terms. The wrapper fetches those inputs from their publishers. [Upstream notices](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/NOTICE)
