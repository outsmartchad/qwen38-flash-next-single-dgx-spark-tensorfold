import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import benchmark
import local_api
import recipe


class LaunchTests(unittest.TestCase):
    image_id = "sha256:" + "a" * 64
    def test_plan_matches_start_and_ignores_parent_launch_settings(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "HOST": "0.0.0.0", "CONTEXT": "16", "DOCKER_HOST": "tcp://example.invalid:2375",
            "VISION_URLS": "1", "TF_FLASHNEXT_DRAFTS": "1", "BASH_ENV": "/example.invalid",
        }):
            directory = Path(temp) / "new data"
            action = recipe.Recipe(directory, None, 18889)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                action.plan()
            self.assertFalse(directory.exists())
            self.assertIsNone(json.loads(out.getvalue())["settings"]["IMAGE"])
            action.state.mkdir(parents=True)
            (action.state / "prepared").write_text(
                f"model={recipe.MODEL_ID}@{recipe.MODEL_REVISION} "
                f"image={self.image_id} patches={recipe.PATCH_HASH}\n"
            )
            with contextlib.redirect_stdout(io.StringIO()) as prepared_plan:
                action.plan()
            plan = json.loads(prepared_plan.getvalue())
            self.assertEqual(plan["settings"]["IMAGE"], self.image_id)
            self.assertEqual(plan["endpoint"], "http://127.0.0.1:18889/v1")
            self.assertEqual(plan["settings"]["VIDEO"], "1")
            self.assertEqual(plan["settings"]["VISION_URLS"], "0")
            self.assertEqual(plan["settings"]["TF_FLASHNEXT_DRAFTS"], "6")
            with (patch.object(action, "inspect_owned", side_effect=[None, {"State": {"Running": True}, "Image": self.image_id}]),
                  patch.object(action, "port_in_use", return_value=False),
                  patch.object(action, "verify_image") as verify_image,
                  patch.object(action, "verify_source"),
                  patch.object(action, "status"),
                  patch.object(action, "run") as command):
                action.start()
            verify_image.assert_called_once_with(self.image_id)
            args, kwargs = command.call_args
            self.assertEqual(args[0], plan["argv"])
            self.assertEqual(kwargs["cwd"], Path(plan["source_dir"]))
            self.assertEqual({key: kwargs["env"][key] for key in plan["settings"]},
                             plan["settings"])
            self.assertEqual(kwargs["env"]["DOCKER_HOST"], "unix:///var/run/docker.sock")
            self.assertNotIn("BASH_ENV", kwargs["env"])

    def test_occupied_port_refuses_start(self):
        with tempfile.TemporaryDirectory() as temp:
            action = recipe.Recipe(Path(temp), None, 18889)
            with (patch.object(action, "inspect_owned", return_value=None),
                  patch.object(action, "port_in_use", return_value=True),
                  patch.object(action, "upstream") as upstream):
                with self.assertRaisesRegex(recipe.RecipeError, "already in use"):
                    action.start()
                upstream.assert_not_called()

    def test_stop_refuses_name_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            action = recipe.Recipe(Path(temp), None, 18889)
            foreign = {"Id": "123", "Name": f"/{action.container}", "Config": {"Image": "foreign"}}
            with (patch.object(action, "run", side_effect=["123\n", json.dumps([foreign])]),
                  patch.object(action, "upstream") as upstream):
                with self.assertRaisesRegex(recipe.RecipeError, "different launch"):
                    action.stop()
                upstream.assert_not_called()

    def test_stop_uses_immutable_id_and_state_directories_use_distinct_names(self):
        with tempfile.TemporaryDirectory() as temp:
            action = recipe.Recipe(Path(temp) / "one", None, 18889)
            second = recipe.Recipe(Path(temp) / "two", None, 18889)
            self.assertNotEqual(action.container, second.container)
            owned = {"Id": "immutable-id", "State": {"Running": True}}
            with (patch.object(action, "inspect_owned", return_value=owned),
                  patch.object(action, "run") as command,
                  contextlib.redirect_stdout(io.StringIO())):
                action.stop()
            self.assertEqual([args[0][-1] for args, _kwargs in command.call_args_list],
                             ["immutable-id", "immutable-id"])

    def test_prepare_refuses_mounting_a_credential_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            action = recipe.Recipe(Path(temp), None, 18889)
            action.hf_cache.mkdir()
            (action.hf_cache / "stored_tokens").symlink_to(action.hf_cache / "absent")
            with patch.object(action, "upstream") as upstream:
                with self.assertRaisesRegex(recipe.RecipeError, "contains a credential entry"):
                    action.prepare()
                upstream.assert_not_called()

    def test_verify_image_uses_immutable_id_and_checks_patch_label(self):
        with tempfile.TemporaryDirectory() as temp:
            action = recipe.Recipe(Path(temp), None, 18889)
            with patch.object(action, "run", return_value=f"{self.image_id} wrong_hash embedded") as inspect:
                with self.assertRaisesRegex(recipe.RecipeError, "labels do not match"):
                    action.verify_image(self.image_id)
                self.assertEqual(inspect.call_args.args[0][-1], self.image_id)


class FakeApi(BaseHTTPRequestHandler):
    def log_message(self, *_parts):
        pass

    def send_data(self, body, mime_type):
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, body):
        self.send_data(json.dumps(body).encode(), "application/json")

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        elif self.path == "/v1/models":
            self.send_json({"data": [{"id": local_api.MODEL}]})
        elif self.path == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b" ")
                    time.sleep(0.01)
            except (ConnectionError, OSError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][0]["content"]
        count = 33 + 20 * sum(row.startswith("record ") for row in text.splitlines())
        if self.path == "/tokenize":
            self.send_json({"count": count})
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not body.get("stream"):
            self.send_json({"choices": [{"message": {"content": "SPARK_OK"}}]})
            return
        marker = re.search(r"AUDIT CODE: (FROST-[A-F0-9]+)", text)
        answer = marker[1] if marker else "An inventory record"
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": answer[:3]}}]},
            {"choices": [{"delta": {"content": answer[3:]}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "tensorfold": {"prefill_seconds": 0.2, "decode_seconds": 0.1}},
        ]
        if text != "omit counters":
            events.append({"usage": {"prompt_tokens": count, "completion_tokens": 4,
                                     "prompt_tokens_details": {"cached_tokens": 1}}})
        data = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        self.send_data((data + "data: [DONE]\n\n").encode(), "text/event-stream")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer((local_api.HOST, 0), FakeApi)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_status_and_smoke(self):
        action = recipe.Recipe(Path("/tmp/spark-test"), None, self.port)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            action.status()
        self.assertEqual(json.loads(out.getvalue())["models"], [local_api.MODEL])
        with contextlib.redirect_stdout(io.StringIO()) as smoke:
            action.smoke()
        self.assertIn("returned SPARK_OK", smoke.getvalue())

    def test_quick_benchmark_uses_usage_and_checks_the_needle(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = benchmark.benchmark(self.port, "quick", 123, 3)
        self.assertEqual([row["streams"] for row in result["completion_batches"]], [1, 5])
        self.assertEqual(len(result["long_prompts"]), 1)
        row = result["long_prompts"][0]
        self.assertTrue(row["audit_code_matched"])
        self.assertEqual(row["prompt_tokens"], row["tokenized_prompt_tokens"])
        self.assertEqual(row["completion_tokens"], 4)
        self.assertEqual(row["engine_decode_tokens_per_second"], 40)
        self.assertEqual(row["cached_prompt_tokens"], 1)
        self.assertGreaterEqual(row["elapsed_seconds"], row["first_content_seconds"])

    def test_stream_needs_explicit_counters(self):
        with self.assertRaisesRegex(local_api.ApiError, "explicit positive"):
            benchmark.stream_completion(self.port, "omit counters", 16, 3)

    def test_wall_timeout_on_a_trickling_response(self):
        begin = time.monotonic()
        with self.assertRaisesRegex(local_api.ApiError, "timed out"):
            local_api.json_request(self.port, "/slow", timeout=0.15)
        self.assertLess(time.monotonic() - begin, 0.8)

    def test_prompt_is_seeded(self):
        self.assertEqual(benchmark.long_prompt(12, 100), benchmark.long_prompt(12, 100))
        self.assertNotEqual(benchmark.long_prompt(12, 100), benchmark.long_prompt(13, 100))


if __name__ == "__main__":
    unittest.main()
