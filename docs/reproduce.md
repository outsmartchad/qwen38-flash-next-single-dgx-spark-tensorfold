# Reproduce the recipe

[Quick start](../README.md#quick-start) · [How it works](how-it-works.md) · [Troubleshooting](troubleshooting.md) · [Results and raw data](../results/README.md)

## Before you start

Run the install and benchmark commands on the Spark. An SSH shell into the Spark works for both. The reference machine was one NVIDIA GB10 with 128 GB of advertised unified memory, Ubuntu 24.04.5 LTS, NVIDIA driver 580.178.04, and Docker 29.6.2. [Recorded host details](../results/running-profile.json)

- **Host tools** · Python 3.9 or later, Bash 4.4 or later, Git, curl, and the standard Linux utilities. If `preflight` reports a missing `ss`, install Ubuntu’s `iproute2` package. `flock` comes from `util-linux`, and `readlink`, `sha256sum`, and `timeout` come from `coreutils`.
- **Docker and GPU** · `nvidia-smi -L` should show one GB10. `docker info` should work as your current user, show an `nvidia` runtime, and be backed by the Docker Engine at `/var/run/docker.sock`. [Docker post-install steps](https://docs.docker.com/engine/install/linux-postinstall/) · [NVIDIA Container Toolkit setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Memory** · Start with at least 107 GiB in Linux `MemAvailable` and close other GPU workloads. The limit applies before startup. Expect available memory to fall once the model has loaded.
- **Storage** · Allow around 160 GiB for a fresh installation when Docker and the weight cache share a filesystem. If they use separate mounts, reserve about 115 GiB under the weight cache and 40 GiB under Docker’s storage root. The pinned upstream preparation script checks both mounts before starting the build. If Docker uses the containerd image store, check that filesystem too. [Docker image storage](https://docs.docker.com/engine/storage/containerd/) · [Upstream disk checks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/scripts/prepare.sh#L71-L91)
- **Downloads** · `fetch` and `prepare` need internet access to GitHub, NVIDIA NGC, Hugging Face, and the pinned dependency hosts used during the build. Preparation can use an installed `hf` CLI or download from inside the build image. Use a dedicated Hugging Face cache for these public weights. [Upstream build and download steps](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9/scripts/prepare.sh)

```sh
uname -m
nvidia-smi -L
docker --host unix:///var/run/docker.sock info --format '{{json .Runtimes}}'
docker_root="$(docker --host unix:///var/run/docker.sock info --format '{{.DockerRootDir}}')"
df -h "$HOME" "$docker_root"
awk '/^MemAvailable:/ {printf "%.1f GiB available\n", $2/1048576}' /proc/meminfo
```

On a stock DGX Spark, NVIDIA includes the GPU container runtime. To test Docker GPU access before the model build, run NVIDIA’s sample command. It can pull an additional CUDA image. [DGX Spark Docker documentation](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)

```sh
docker --host unix:///var/run/docker.sock run --rm --gpus=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

## Install and verify

Follow the [quick start](../README.md#quick-start). `preflight` checks the host and free port, `fetch` gets the pinned upstream source, and `prepare` builds the image and stages the checkpoint. Build and download time depend on the Spark’s network and cache. Start with the default loopback port, 18889.

Use `python3 bin/recipe.py plan` before preparing if you want to see the model revision, build tag, paths, and serving settings. After launch, `status` prints the health response and lists `Qwen3.8-Flash-Next`. Use `ss -ltn 'sport = :18889'` to confirm the listening socket is on `127.0.0.1`. The text smoke test must return `SPARK_OK`. The optional media test checks for red in the PNG and red → blue → green in the MP4.

### Choose a different cache or port

The default data directory is `$HOME/.local/share/dgx-spark-flash-next-recipe`. It stores the pinned source, build state, kernel cache, and the default Hugging Face cache. You can use a writable volume with a larger storage budget. In Bash, use the same argument array across commands.

```bash
recipe_args=(--data-dir "$HOME/.local/share/my-flash-next" --hf-cache "$HOME/.cache/flash-next-public" --port 18890)
python3 bin/recipe.py preflight "${recipe_args[@]}"
python3 bin/recipe.py fetch "${recipe_args[@]}"
python3 bin/recipe.py prepare "${recipe_args[@]}"
python3 bin/recipe.py start "${recipe_args[@]}"
python3 bin/recipe.py status "${recipe_args[@]}"
python3 bin/recipe.py smoke "${recipe_args[@]}"
python3 tools/multimodal_smoke.py --port 18890
python3 bin/recipe.py stop "${recipe_args[@]}"
```

Set `--hf-cache` to the cache **root** that contains the `hub/` directory. If it contains an HF token file, choose a separate public cache. Use `--port 18890` on benchmark commands as well if you picked that port.

## Connect an API client

On the Spark, send an OpenAI-style request to its loopback API.

```sh
curl --noproxy '*' -fsS --max-time 60 http://127.0.0.1:18889/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Explain unified memory in two sentences."}],"max_tokens":160,"chat_template_kwargs":{"enable_thinking":false}}'
```

Use `http://127.0.0.1:18889/v1` as the base URL and `Qwen3.8-Flash-Next` as the model. If the client requires a key field, a placeholder such as `local` is sufficient for this loopback server.

For a client on your laptop, replace `your-user@spark-host` with your normal Spark SSH destination. Leave this running in a separate terminal, then use the same base URL in your client.

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18889:127.0.0.1:18889 your-user@spark-host
```

## Run the benchmarks

Run from the recipe root on the Spark with the model already up and other clients idle. Start with the inventory check, then run the published workload you want to compare. Pick new output names for re-runs.

```sh
mkdir -p data/runs
python3 bin/benchmark.py --profile quick --output data/runs/quick.json
python3 bin/benchmark.py --profile prose-thinking-on --output data/runs/prose-thinking-on.json
python3 bin/benchmark.py --profile full-decode --sample-memory --seed 2026112041 --output data/runs/decode-memory.json
python3 bin/benchmark.py --profile full-prefill --seed 2026112032 --output data/runs/full-prefill.json
```

The quick profile uses synthetic inventory prompts and a roughly 4K retrieval check. The thinking-on profile exercises the same hash-map prose measurement protocol as the 56.0 tok/s headline. Compare the separate whole-batch thinking-off decode and cold-prefill runs under their own definitions. When you rerun `full-prefill` on the resident server, choose a fresh `--seed` for cold inputs. That changes the input text, so use the reported prompt token counts when comparing.

[Sampling, token budgets, timing definitions, and targeted engine checks](benchmarking.md) · [Published results](../results/README.md)

## Run the local tests

```sh
python3 -m unittest discover -s tests -v
```

The tests check launcher arguments and exercise the clients against a mock server on an ephemeral loopback port. They run without Docker, model weights, or a GPU.

## Stop and restart

```sh
python3 bin/recipe.py stop
python3 bin/recipe.py start
python3 bin/recipe.py smoke
```

You can leave the server stopped after the first command. `stop` removes only the managed Docker container and retains the weights, build image, and kernel cache. Check logs before stopping a failed launch if you want to save the error.
