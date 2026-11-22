# Troubleshooting

[Quick start](../README.md#quick-start) · [Reproduce](reproduce.md) · [How it works](how-it-works.md)

## Preflight fails

- **Wrong architecture or GPU** · Run on the Spark and confirm `uname -m` reports `aarch64` and `nvidia-smi -L` shows one GB10. Commands run remotely through SSH still execute on the Spark.
- **Docker socket permission denied** · Check `docker --host unix:///var/run/docker.sock info` as the account that runs the recipe. Follow Docker’s post-install instructions and reconnect your SSH session if your Docker group membership changed. Docker access grants root-level privileges. [Docker post-install setup](https://docs.docker.com/engine/install/linux-postinstall/)
- **Docker says it cannot select a GPU device** · Confirm the driver works on the host and run NVIDIA’s sample workload with `--gpus all`. The runtime list can omit `nvidia` even when container GPU access works. [Toolkit setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) · [GPU sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)
- **Available memory below 107 GiB** · Stop other inference and media workloads, check `nvidia-smi`, and rerun preflight. Check `MemAvailable` in `/proc/meminfo` again if a workload has just stopped.
- **Port 18889 is busy** · Identify the listener with `ss -ltnp 'sport = :18889'`. If it already serves the intended model, compare its launch settings with this recipe before benchmarking. Otherwise use a different free `--port` consistently across recipe and client commands. [Port and path example](reproduce.md#choose-a-different-cache-or-port)

## Preparation fails

- **No space left** · Check both the weight cache filesystem and Docker’s storage filesystem. Moving `--hf-cache` changes where weights download. Docker image layers still use Docker’s storage. [Disk checks](reproduce.md#before-you-start)
- **An image layer or weight shard failed to download** · Check network access to the failing host in the build output and retry `prepare` with the same paths. The download reuses completed files. If a host `hf` command fails, repair or update that CLI. On hosts without `hf`, the preparation script downloads from inside the build image. For permission trouble under the cache, choose a separate directory that your Spark account owns.
- **Prepared image missing, checkpoint missing, or marker mismatch** · Run `prepare` against the intended data directory and cache. Continue with the same custom switches if you use them.
- **Pinned source reports local changes** · The source directory is a verified build input. Keep the local changes and run `fetch` and `prepare` with a fresh `--data-dir`. You can pass the prior public cache root with `--hf-cache` to reuse its weights.
- **Another command holds the recipe lock** · Wait for the other prepare, start, or stop command to finish. It may still be downloading, building, or loading the checkpoint.

## Server fails to start or answer

Inspect the expected container and its final log lines. If you override the port or paths, pass those same options to `plan`.

```sh
container="$(python3 bin/recipe.py plan | python3 -c 'import json, sys; print(json.load(sys.stdin)["settings"]["CONTAINER_NAME"])')"
docker --host unix:///var/run/docker.sock ps -a --filter "name=^/${container}$"
docker --host unix:///var/run/docker.sock logs --tail 80 "$container"
```

- **Startup timed out while the container still loads** · Check those logs and run `status` before starting it again. When launching again after the previous container has stopped, `start --wait-timeout 3600` allows up to an hour.
- **Container exited or CUDA allocation failed** · Check the logs and available memory. With other inference clients stopped, retry using the default profile after freeing memory. `stop` removes the recipe container and retains its cache for the next load.
- **Laptop gets connection refused** · Run `status` and `smoke` on the Spark first. If they pass, check that the SSH forwarding process is still running, then use the forwarded port on the laptop. [SSH example](reproduce.md#connect-an-api-client)
- **Chat reports an unknown model** · Set your client’s model ID to `Qwen3.8-Flash-Next` and check `/v1/models` on the same endpoint.
- **Image or video request fails** · Run the bundled media smoke test first. The media parts take inline `data:` URLs. A failing fixture gives you a small reproducible input to report.

## Benchmark results differ

Compare the same prompt, thinking mode, sampler, completion budget, and concurrency. The 56.0 tok/s figure uses the best measured one-stream thinking-on prose wave and excludes TTFT from decode speed. Cache reuse is recorded per request. The whole-batch thinking-off rate uses a different timing window. Run one benchmark at a time against the local server. [Benchmark methodology](benchmarking.md)

For a bug report, include the recipe commit, `uname -m`, driver and Docker versions, the exact failing command, and the relevant log excerpt. Review the excerpt for credentials, personal paths, and host details before publishing it.
