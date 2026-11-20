#!/usr/bin/env python3
"""Check the running localhost TensorFold API with known synthetic media."""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MODEL = "Qwen3.8-Flash-Next"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
COLORS = re.compile(r"\b(red|blue|cyan|green|yellow|orange|purple|magenta|black|white)\b", re.I)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def check_media(opener, port, name, mime, question, expected):
    fixture = (FIXTURES / name).read_bytes()
    attachment = base64.b64encode(fixture).decode("ascii")
    field = "video_url" if mime.startswith("video/") else "image_url"
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": field, field: {"url": f"data:{mime};base64,{attachment}"}},
                {"type": "text", "text": question},
            ],
        }],
        "max_tokens": 100,
        "reasoning_effort": "none",
        "temperature": 0,
    }
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    began = time.monotonic()
    with opener.open(request, timeout=90) as response:
        result = json.loads(response.read(1024 * 1024))
    elapsed = time.monotonic() - began
    if result.get("model") != MODEL:
        raise ValueError("The endpoint reported a different model.")
    answer = result["choices"][0]["message"]["content"]
    if not isinstance(answer, str):
        raise ValueError(f"{field} returned no answer text.")

    sequence = []
    for color in COLORS.findall(answer.lower()):
        color = "blue" if color == "cyan" else color
        if not sequence or sequence[-1] != color:
            sequence.append(color)
    if sequence != expected:
        raise ValueError(f"{field} expected {expected}, observed {sequence}.")

    return {
        "fixture": name,
        "content_part": field,
        "colors": sequence,
        "elapsed_s": round(elapsed, 3),
        "usage": {key: result["usage"][key] for key in ("prompt_tokens", "completion_tokens")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18889)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        still = check_media(
            opener, args.port, "red-square.png", "image/png",
            "What color is the square? Reply with one color word.",
            ["red"],
        )
        clip = check_media(
            opener, args.port, "color-cuts.mp4", "video/mp4",
            "List the square's colors in chronological order. Reply with only the color names.",
            ["red", "blue", "green"],
        )
    except (OSError, URLError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Multimodal smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"model": MODEL, "checks": [still, clip]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
