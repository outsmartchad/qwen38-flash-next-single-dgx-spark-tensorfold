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
git clone https://github.com/outsmartchad/dgx-spark-flash-next-recipe.git
cd dgx-spark-flash-next-recipe

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

```sh
python3 bin/benchmark.py --profile quick --output quick.json
```

Run from the Spark while other inference clients are idle. The quick profile checks one and five simultaneous streams, then a roughly 4K-token retrieval prompt. Larger profiles exercise longer prompts up to 200K and take longer to run. Each JSON record captures reported token counts, first content latency, decode or completion rates, aggregate throughput where measured, and whether the model recovered the synthetic marker. See [the benchmark methodology](docs/benchmarking.md) before comparing numbers.

### Observed on one Spark

Two seeded runs on the resident server measured **47.4–59.2 tok/s** in single-stream engine decode and **134.6–135.8 tok/s** in end-to-end throughput across five short streams. These are individual runs with synthetic inventory output. [Quick run JSON](results/quick-2026-11-20.json) · [Standard run JSON](results/standard-2026-11-20.json)

The standard run found the exact marker in **4,090**, **32,787**, and **131,056** input tokens. Time to first content was **1.80 s**, **14.01 s**, and **59.66 s**, respectively, with one request at a time.

The launch settings expose a maximum context and configured slot count. The attached run data bounds the workloads we actually tested. A clean-machine install and independent reproduction are still open, and the recorded local image ID is an observation from a built image.

## Build another recipe

The [single-Spark backlog](docs/model-backlog.md) records the candidate checkpoints, engines, sources, and hardware evidence. It includes the 27B comparator, alternate Flash Next serving paths, and larger-model candidates that need compression or offload. [Recipe template](docs/reusing-this-recipe.md)

If you want to wrap the server with Telegram, voice, mesh generation, and video generation, see the [DGX Spark multimodal setup](https://github.com/outsmartchad/dgx-spark-setup).

## Credits and license

Thanks to [Mia’s AI Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) for the upstream recipe and patches, [TensorFold contributors](https://github.com/ashhart/TensorFold) for the engine, [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) for the quantized checkpoint, and [the Qwen team](https://github.com/QwenLM) for the model family. [Sybil Solutions](https://github.com/sybil-solutions) informed the reproducibility and benchmark format.

Our scripts, documentation, and synthetic fixtures are [MIT licensed](LICENSE). The upstream build, NGC image, and model retain their own terms. The wrapper fetches those inputs from their publishers. [Upstream notices](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/NOTICE)
