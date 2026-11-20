#!/usr/bin/env python3
"""Launch the pinned upstream recipe on one Linux ARM64 DGX Spark."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

from local_api import ApiError, CONTEXT, DEFAULT_PORT, HOST, MODEL, PARALLEL
from local_api import json_request, port_number, require_model

UPSTREAM_REPO = "https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git"
UPSTREAM_COMMIT = "b956dba8a468f1cf376d60bbbd9b9ed51ff2f0a9"
TENSORFOLD_COMMIT = "93a2e93ecebff8aa52b88e58c5dd222516e69c5f"
BASE_IMAGE = "nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
MODEL_ID = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
MODEL_REVISION = "b9ce18a6887e18c089fdea8509a33e63429a3300"
IMAGE = f"tensorfold-qwen38-recipe:v1.2.7-{TENSORFOLD_COMMIT[:7]}-{UPSTREAM_COMMIT[:8]}"
PATCH_HASH = "261c0ede7fce"  # upstream image_hash with the pinned BASE_IMAGE
IMAGE_ID_PATTERN = r"sha256:[0-9a-f]{64}"
MIN_START_GIB = 107
WAIT_SECONDS = 1800
GIB = 1024 ** 3
REQUIRED_TOOLS = ("bash", "curl", "docker", "flock", "git", "nvidia-smi",
                  "python3", "readlink", "sha256sum", "ss", "timeout")


class RecipeError(RuntimeError):
    pass


class Recipe:
    def __init__(self, data_dir, hf_cache, port, wait_timeout=WAIT_SECONDS):
        self.data_dir = data_dir.expanduser().resolve()
        self.hf_cache = hf_cache.expanduser().resolve() if hf_cache else self.data_dir / "huggingface"
        self.port = port
        self.source = self.data_dir / f"upstream-{UPSTREAM_COMMIT}"
        self.kernel_cache = self.data_dir / "kernels"
        self.state = self.data_dir / "state"
        data_id = hashlib.sha256(os.fsencode(self.data_dir)).hexdigest()[:12]
        self.container = f"dgx-flash-next-recipe-{port}-{data_id}"
        self.wait_timeout = wait_timeout
        if ":" in str(self.data_dir) or ":" in str(self.hf_cache):
            raise RecipeError("Docker bind mount paths cannot contain a colon")
        self.base_env = {
            key: os.environ[key] for key in
            ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR",
             "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE")
            if key in os.environ
        }
        self.base_env.update(
            DOCKER_HOST="unix:///var/run/docker.sock",
            NO_PROXY="localhost,127.0.0.1",
            GIT_TERMINAL_PROMPT="0",
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
        )
        self.settings = {
            "HOST": HOST, "PORT": str(port), "SERVED_NAME": MODEL,
            "CHECKPOINT": "nvfp4", "MODEL_ID": MODEL_ID, "MODEL_REVISION": MODEL_REVISION,
            "TF_REF": TENSORFOLD_COMMIT, "BASE_IMAGE": BASE_IMAGE, "IMAGE": IMAGE,
            "CONTAINER_NAME": self.container,
            "CONTEXT": str(CONTEXT), "PARALLEL": str(PARALLEL), "KV_DTYPE": "fp8",
            "VISION": "1", "VIDEO": "1", "VISION_URLS": "0", "PDF": "0", "ABLIT": "0",
            "DRAFTS": "1", "TF_FLASHNEXT_CHUNK": "3072", "TF_FLASHNEXT_DRAFTS": "6",
            "TF_FLASHNEXT_CONFIDENCE": "-0.4", "TF_FLASHNEXT_SHARED_DRAFTS": "0",
            "TENSORFOLD_MEMORY_RESERVE_GIB": "9", "TENSORFOLD_VIDEO_TOKENS": "16384",
            "PREPARE": "0", "PULL": "0", "WAIT_TIMEOUT": str(wait_timeout),
            "HF_CACHE": str(self.hf_cache), "HF_HOME": str(self.hf_cache),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_TOKEN_PATH": os.devnull,
            "KERNEL_CACHE": str(self.kernel_cache), "STATE_DIR": str(self.state),
            "LOG_DIR": str(self.data_dir / "logs"),
        }

    def run(self, argv, *, timeout=10, cwd=None, env=None, capture=True):
        return subprocess.run(
            argv, cwd=cwd, env=self.base_env if env is None else env,
            timeout=timeout, check=True, text=True, capture_output=capture,
        ).stdout

    def git(self, directory, *args, timeout=20):
        return self.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(directory), *args],
            timeout=timeout,
        ).strip()

    @contextlib.contextmanager
    def lock(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / "recipe.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RecipeError("another recipe command holds the data directory lock") from error
            yield

    def verify_source(self, directory):
        if not (directory / ".git").is_dir():
            raise RecipeError(f"Pinned source missing at {directory}. Run fetch.")
        if self.git(directory, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
            raise RecipeError(f"unexpected commit in {directory}")
        if self.git(directory, "status", "--porcelain", "--untracked-files=all", "--ignored"):
            raise RecipeError(f"unexpected local files or changes in {directory}")

    def fetch(self):
        if self.source.exists():
            self.verify_source(self.source)
            return
        with tempfile.TemporaryDirectory(prefix="checkout-", dir=self.data_dir) as temporary:
            directory = Path(temporary) / "source"
            directory.mkdir()
            self.git(directory, "init", "--quiet")
            self.git(directory, "fetch", "--depth=1", "--no-tags", UPSTREAM_REPO, UPSTREAM_COMMIT,
                     timeout=180)
            if self.git(directory, "rev-parse", "FETCH_HEAD") != UPSTREAM_COMMIT:
                raise RecipeError("upstream fetch returned an unexpected commit")
            self.git(directory, "checkout", "--quiet", "--detach", UPSTREAM_COMMIT)
            self.verify_source(directory)
            directory.rename(self.source)
        print(f"Fetched pinned upstream source at {self.source}")

    def verify_public_cache(self):
        for filename in ("token", "stored_tokens"):
            if os.path.lexists(self.hf_cache / filename):
                raise RecipeError("Hugging Face cache contains a credential entry. Select a public cache root.")

    def prepared_image_id(self, *, required):
        try:
            marker = (self.state / "prepared").read_text().strip()
        except FileNotFoundError as error:
            if required:
                raise RecipeError("Prepared image record is missing. Run prepare.") from error
            return None
        model = re.escape(f"{MODEL_ID}@{MODEL_REVISION}")
        match = re.fullmatch(f"model={model} image=({IMAGE_ID_PATTERN}) patches={PATCH_HASH}", marker)
        if not match:
            raise RecipeError("Prepared model or image record does not match this recipe. Run prepare.")
        return match[1]

    def verify_image(self, image_id):
        try:
            fields = self.run([
                "docker", "image", "inspect", "--format",
                '{{.Id}} {{index .Config.Labels "tf.patches"}} {{index .Config.Labels "tf.kernels"}}',
                image_id,
            ]).split()
        except subprocess.CalledProcessError as error:
            raise RecipeError("Prepared image is unavailable in local Docker. Run prepare.") from error
        if fields != [image_id, PATCH_HASH, "embedded"]:
            raise RecipeError("Prepared Docker image labels do not match this recipe. Run prepare.")

    def upstream(self, target, timeout, *, image=IMAGE):
        self.verify_source(self.source)
        env = {**self.base_env, **self.settings, "IMAGE": image}
        self.run(["bash", target], cwd=self.source, env=env, capture=False, timeout=timeout)

    def plan(self):
        print(json.dumps({
            "upstream": UPSTREAM_REPO, "upstream_commit": UPSTREAM_COMMIT,
            "source_dir": str(self.source), "argv": ["bash", "start.sh"],
            "build_image_tag": IMAGE, "expected_patch_hash": PATCH_HASH,
            "settings": {**self.settings, "IMAGE": self.prepared_image_id(required=False)},
            "endpoint": f"http://{HOST}:{self.port}/v1",
        }, indent=2))

    def port_in_use(self):
        return bool(self.run(["ss", "-H", "-ltn", f"sport = :{self.port}"]).strip())

    def preflight(self):
        if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
            raise RecipeError("build and launch require Linux ARM64 on an NVIDIA GB10 system")
        missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
        if missing:
            raise RecipeError(f"required commands missing: {', '.join(missing)}")
        version = self.run([
            "bash", "-c", 'printf "%s.%s\\n" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"',
        ]).strip()
        if tuple(int(part) for part in version.split(".")) < (4, 4):
            raise RecipeError("the pinned upstream scripts require Bash 4.4 or newer")
        gpu = self.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip().splitlines()
        if len(gpu) != 1 or "GB10" not in gpu[0]:
            raise RecipeError("expected exactly one NVIDIA GB10 GPU")
        runtimes = json.loads(self.run(["docker", "info", "--format", "{{json .Runtimes}}"]))
        if not isinstance(runtimes, dict) or "nvidia" not in runtimes:
            raise RecipeError("the local Docker daemon does not list the nvidia runtime")
        memory = {}
        for row in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = row.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.split()[0]) * 1024
        print(f"GPU: {gpu[0]}")
        print(f"Unified memory: {memory['MemAvailable'] / GIB:.1f} GiB available of "
              f"{memory['MemTotal'] / GIB:.1f} GiB, upstream startup threshold {MIN_START_GIB} GiB")
        for location in dict.fromkeys((self.data_dir, self.hf_cache)):
            existing = next(path for path in (location, *location.parents) if path.exists())
            print(f"Disk: {shutil.disk_usage(existing).free / GIB:.1f} GiB free for {location}")
        in_use = self.port_in_use()
        print(f"TCP port {self.port}: {'in use' if in_use else 'free'}")
        if memory["MemAvailable"] < MIN_START_GIB * GIB or in_use:
            raise RecipeError("startup requires at least 107 GiB MemAvailable and a free TCP port")

    def inspect_owned(self):
        container_id = self.run([
            "docker", "container", "ls", "--all", "--filter", f"name=^/{self.container}$",
            "--no-trunc", "--format", "{{.ID}}",
        ]).strip()
        if not container_id:
            return None
        records = json.loads(self.run(["docker", "container", "inspect", container_id]))
        if len(records) != 1 or records[0].get("Id") != container_id:
            raise RecipeError(f"ambiguous Docker inspection for {self.container}")
        record = records[0]
        cfg = record["Config"]
        command = cfg.get("Cmd") or []
        serve = command[command.index("tensorfold-native"):] if "tensorfold-native" in command else []
        model_path = f"/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/{MODEL_REVISION}"
        expected = ["tensorfold-native", "serve", model_path, "--name", MODEL,
                    "--host", HOST, "--port", str(self.port), "--context", str(CONTEXT),
                    "--parallel", str(PARALLEL)]
        cache_mounts = any(
            mount.get("Destination") == "/cache"
            and Path(mount.get("Source", "/")) == self.kernel_cache / PATCH_HASH
            for mount in record.get("Mounts", [])
        )
        required_flags = (("--kv-cache", "fp8"), ("--vision-video-tokens", "16384"))
        switches = set(zip(serve, serve[1:]))
        image_id = record.get("Image") or ""
        if (record.get("Name") != f"/{self.container}" or not re.fullmatch(IMAGE_ID_PATTERN, image_id)
                or cfg.get("Image") != image_id
                or serve[:len(expected)] != expected or not cache_mounts
                or "--vision" not in serve or not set(required_flags).issubset(switches)
                or record["HostConfig"].get("NetworkMode") != "host"):
            raise RecipeError(f"container name {self.container} is in use by a different launch")
        return record

    def prepare(self):
        self.verify_public_cache()
        self.upstream("scripts/prepare.sh", timeout=8 * 3600)
        self.verify_image(self.prepared_image_id(required=True))

    def start(self):
        existing = self.inspect_owned()
        running = existing and existing["State"]["Running"]
        if not running and self.port_in_use():
            raise RecipeError(f"TCP port {self.port} is already in use")
        self.verify_public_cache()
        image_id = self.prepared_image_id(required=True)
        self.verify_image(image_id)
        if running:
            if existing["Image"] != image_id:
                raise RecipeError("The running container uses an earlier image. Stop it before starting this image.")
            self.status()
            return
        self.upstream("start.sh", timeout=self.wait_timeout + 240, image=image_id)
        launched = self.inspect_owned()
        if not launched or not launched["State"]["Running"] or launched["Image"] != image_id:
            raise RecipeError(f"{self.container} is no longer running. Inspect its Docker logs.")
        self.status()

    def stop(self):
        owned = self.inspect_owned()
        if not owned:
            print(f"No recipe container named {self.container}")
            return
        if owned["State"]["Running"]:
            self.run(["docker", "container", "stop", "--time", "30", owned["Id"]], timeout=45)
        self.run(["docker", "container", "rm", owned["Id"]])
        print(f"Stopped and removed {self.container}")

    def status(self):
        reply = json_request(self.port, "/health")
        models = require_model(self.port)
        print(json.dumps({"endpoint": f"http://{HOST}:{self.port}/v1", "health": reply,
                          "models": models}, indent=2))

    def smoke(self):
        require_model(self.port)
        body = {"model": MODEL, "temperature": 0, "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Reply with exactly SPARK_OK and no other text."}]}
        result = json_request(self.port, "/v1/chat/completions", body, timeout=60)
        try:
            content = result["choices"][0]["message"]["content"].strip()
        except (IndexError, KeyError, TypeError, AttributeError) as error:
            raise RecipeError("completion has no text") from error
        if content != "SPARK_OK":
            raise RecipeError(f"unexpected smoke reply: {content[:120]!r}")
        print(f"{MODEL} returned SPARK_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "fetch", "prepare", "start",
                                            "status", "smoke", "stop"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / ".local" / "share" / "dgx-spark-flash-next-recipe")
    parser.add_argument("--hf-cache", type=Path, help="optional existing Hugging Face cache root")
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT)
    parser.add_argument("--wait-timeout", type=int, default=WAIT_SECONDS, metavar="SECONDS")
    args = parser.parse_args()
    if not 30 <= args.wait_timeout <= 3600:
        parser.error("--wait-timeout must be 30–3600 seconds")
    try:
        action = Recipe(args.data_dir, args.hf_cache, args.port, args.wait_timeout)
        command = getattr(action, args.command)
        if args.command in {"fetch", "prepare", "start", "stop"}:
            with action.lock():
                command()
        else:
            command()
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        sys.exit(f"{args.command} failed (exit {error.returncode})" + (f": {detail}" if detail else ""))
    except subprocess.TimeoutExpired as error:
        sys.exit(f"{args.command} timed out after {error.timeout} seconds. Check status before retrying.")
    except (RecipeError, ApiError, OSError, ValueError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
