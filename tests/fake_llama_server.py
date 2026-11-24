"""A fake ``llama-server`` for the tests of the test program: same endpoints and fields, no model.

Behaviour is controlled by environment variables:

- FAKE_SLEEP_ON: if the claim contains this text, the answer comes after 30 s (a timeout);
- FAKE_CRASH_FILE: if the file exists, the first request deletes it and kills the process;
- FAKE_REPLY: "json" (default), "prose" (an answer in prose) or "garbage";
- FAKE_NO_GPU: ``--list-devices`` shows no device, like a user without access to the GPU.
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int)
ap.add_argument("--host")
ap.add_argument("-m")
ap.add_argument("--list-devices", action="store_true")
args, _ = ap.parse_known_args()
if args.list_devices:
    print("Available devices:\n  " + ("(none)" if os.environ.get("FAKE_NO_GPU") else "FakeGPU0: fake"))
    sys.exit(0)
LABELS = ("TRUE", "FALSE", "UNKNOWN")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/props":
            self._send({"build_info": "fake", "model_path": args.m, "total_slots": 1})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self._send({"tokens": list(range(len(payload["content"].split())))})
            return
        if self.path == "/v1/systemone":  # decision models: a choice and its probabilities per question
            answers = {}
            for qid, q in payload["questions"].items():
                assert q["type"] == "choice" and payload["state"].startswith(("KNOWLEDGE BASE\n", "REAL WORLD\n"))
                options = list(q["criteria"])
                claim = q["instructions"].rsplit("CLAIM\n", 1)[1]
                pick = options[sum(map(ord, claim)) % len(options)]
                probs = {o: (0.8 if o == pick else 0.2 / (len(options) - 1)) for o in options}
                answers[qid] = {"choice": pick, "probabilities": probs, "confidence": 0.8}
            self._send({"answers": answers, "usage": {"input_tokens": 120, "output_tokens": 0}})
            return
        crash = os.environ.get("FAKE_CRASH_FILE")
        if crash and os.path.exists(crash):
            os.remove(crash)
            os._exit(1)
        claim = payload["messages"][-1]["content"].rsplit("CLAIM\n", 1)[1]
        if os.environ.get("FAKE_SLEEP_ON") and os.environ["FAKE_SLEEP_ON"] in claim:
            time.sleep(30)
        label = LABELS[sum(map(ord, claim)) % 3]
        reply = os.environ.get("FAKE_REPLY", "json")
        content = {
            "json": json.dumps({"answer": label}),
            "prose": f"The answer is {label}.",
            "garbage": "I am not sure.",
        }[reply]
        self._send(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content, "reasoning_content": "step one step two"},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                "timings": {"predicted_per_second": 50.0, "prompt_per_second": 500.0, "cache_n": 90},
                "echo": {k: payload.get(k) for k in ("seed", "max_tokens", "temperature", "chat_template_kwargs")},
            }
        )


ThreadingHTTPServer((args.host or "127.0.0.1", args.port), Handler).serve_forever()
sys.exit(0)
