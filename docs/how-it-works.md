# How Qwen3.8 Flash Next works on TensorFold

[Reproduce](reproduce.md) · [Troubleshooting](troubleshooting.md) · [Results](../results/README.md)

## How the model works

This recipe uses NVIDIA’s pinned `Qwen3.8-Flash-Next-FP4-Local` snapshot at `b9ce18a6887e18c089fdea8509a33e63429a3300`. Qwen3.8-Flash-Next is a 125B sparse MoE with 6B activated per token, plus 51B of n-gram embedding tables and a 4B MTP module. The NVFP4 snapshot stores weights compactly, while this serving profile keeps the KV cache in FP8. [Qwen model overview](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [pinned NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300)

The 48-layer language trunk repeats twelve four-layer groups. Each group has three Gated DeltaNet layers followed by one Qwen Sparse Attention layer, with an MoE feed-forward sublayer after each mixer. GDN carries a fixed-size recurrent state, while QSA uses a compressed indexer to select 512 micro-blocks or 2,048 tokens. Each MoE has 512 experts and activates 10 routed experts plus one shared expert. [Qwen architecture report, §2](https://arxiv.org/html/2608.30320v1#S2) · [pinned upstream recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9)

Gated Residual widens each residual stream to four branches, uses an element-wise read gate, and applies a scalar write gate per branch. The n-gram embedding layer is inserted at layer 2 and prefetched from host memory, adding capacity with little per-token compute. The MTP module proposes future tokens for target verification. Qwen's reference design reuses the sparse-attention index selections across successive draft steps. [Qwen model overview](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [Qwen architecture report, §2](https://arxiv.org/html/2608.30320v1#S2) · [TensorFold repository](https://github.com/ashhart/TensorFold)

## How this recipe runs it on one DGX Spark

The local flow is

```text
HF snapshot + pinned TensorFold image
              -> prepare patched native runtime
              -> localhost OpenAI-compatible server
              -> text or media request
              -> prefill, KV-cache growth, MTP verification, stream
```

`bin/recipe.py fetch` obtains the pinned upstream recipe source. `prepare` builds TensorFold 1.2.7 with the pinned patches, stages and verifies the checkpoint, and records the Docker image ID. `start` mounts the prepared cache read-only and binds `127.0.0.1:18889`. The profile has five request slots, a 262,144-token window, FP8 KV, MTP with up to six drafts, and a 9 GiB reserve. [Launcher](../bin/recipe.py) · [observed profile](../results/running-profile.json)

The measured decode sweep used one through five short text streams, 400 generated tokens, thinking enabled, temperature 0.7, top-p 1, and top-k 20. Best aggregate rates were 56.0, 82.6, 102.3, 119.9, and 125.6 tokens per second. The separate fresh-prompt sweep used 4,084 through 200,007 input tokens and reached 2,099 tokens per second at 200,007 tokens. These are inference measurements from resident runs. [Benchmark data](../results/decode-prose-thinking-on-2026-11-20.json) · [prefill data](../results/full-prefill-2026-11-20.json)

The tradeoff is visible in the capacity numbers. A five-slot server improves aggregate decode through continuous batching, while each active request competes for KV and unified memory. Long prompts consume prefill time and cache space before the first token. Media requests add vision embeddings and frame work, so their latency and memory use require separate checks. The repository’s validation covers text streaming, context admission, prompt reuse, concurrent greedy parity, one synthetic image, and one synthetic video. [Validation receipts](../results/validation-2026-11-20.json) · [media smoke](../results/native-media-smoke.json)

### Capacity and files on disk

The serving log reported about 4.83 GiB of FP8 KV budget for each full 262,144-token slot. Five such slots account for about 24.2 GiB of startup KV budget. Prompt and requested output share each window. The resident server reported 32.50 GiB of Linux available memory at idle in one sample. Each of those figures describes a different measurement or allocation. [RAM method](benchmarking.md#ram-under-decode-load) · [Observed configuration](../results/running-profile.json)

The data directory holds the pinned checkout under `upstream-<commit>/`, preparation and lock files in `state/`, compiled kernels in `kernels/`, and downloaded weights under `huggingface/hub/` by default. Docker keeps layers and container logs in its own storage. The launcher checks the prepared image labels and immutable ID at startup. The status command checks health and model discovery, and the smoke command checks a text completion. [Setup and cache paths](reproduce.md)
