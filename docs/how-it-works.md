# How the TensorFold recipe works

[Reproduce](reproduce.md) · [Troubleshooting](troubleshooting.md) · [Results](../results/README.md)

## Build and serve

1. The Python launcher fetches an exact [Mia recipe commit](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9).
2. Preparation builds TensorFold 1.2.7 with the pinned upstream patches on NVIDIA’s digest-pinned PyTorch image. The upstream step downloads the public [NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300), checks the shard layout, and records the verified Docker image ID.
3. The launcher checks that image ID and confirmation labels before starting. The TensorFold native server mounts the weight cache read-only and serves the OpenAI-compatible API directly on `127.0.0.1:18889`. Image and inline video inputs use the model’s built-in vision tower.
4. `status` queries health and model discovery. `smoke` checks an actual completion. The dedicated [media check](../fixtures/README.md) uses small synthetic fixtures.

The model uses TensorFold’s Flash Next native path with **NVFP4 quantized expert weights**, **FP8 KV cache**, and **multi-token prediction** with up to six draft tokens per verification round. NVFP4 is a weight format. The key-value cache stores attention state as a request grows. MTP predicts extra tokens and retains the ones the verifier accepts. A prompt that accepts more drafts can decode faster.

[Launcher settings](../bin/recipe.py) · [Observed serving configuration](../results/running-profile.json) · [Pinned upstream build and patch notes](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/README.md)

## Context and memory

The serving profile configures five request slots with a 262,144-token window per request and a 9 GiB memory reserve. The engine reported about 4.83 GiB of startup KV budget per full slot. The input and requested output share the context window. Hot prompt reuse, simultaneous requests, and media inputs all affect the observed resource use and TTFT.

The 1–5 stream decode sweep used short text prompts. The largest prefill measurements each used a single long request. To test several long prompts concurrently, measure that workload separately and check the long-request admission results.

The running system reported 32.50 GiB of available Linux memory at idle for one snapshot. That is system headroom after load. The [memory sampling method](benchmarking.md#ram-under-decode-load) describes the separate decode measurements.

## Files on disk

| Path | Contents |
| --- | --- |
| Repository | Launcher and client scripts, documentation, fixtures, and benchmark data |
| Data directory `upstream-<commit>/` | The pinned, clean Mia checkout |
| Data directory `state/` | Preparation marker, lock files, and launch argument-check output |
| Data directory `kernels/` | TensorFold kernel cache |
| Data directory `huggingface/hub/` | Default model cache with the pinned weights |
| Docker storage root | The base and built images, build layers, and container logs |

A later launch can reuse the prepared image and checkpoint. A different checkpoint, engine version, or serving profile warrants a separate measurement. [Hardware and runtime](../README.md#hardware-and-runtime)
