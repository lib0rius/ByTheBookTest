"""Starting ``llama-server`` (llama.cpp) and sending it requests.

The server runs as a child process of the test program: if the run stops (also when its service unit is
stopped), the server stops too. Requests use ``urllib`` from the standard library.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ServerOptions


class ServerError(RuntimeError):
    """The server does not answer, returned an error, or has stopped."""


class RequestTimeout(RuntimeError):
    """A request took longer than its time limit."""


def _request(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except TimeoutError as e:
        raise RequestTimeout(str(e)) from e
    except urllib.error.HTTPError as e:
        raise ServerError(f"HTTP {e.code}: {e.read()[:500]!r}") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise RequestTimeout(str(e)) from e
        raise ServerError(str(e)) from e
    except (ConnectionError, OSError) as e:
        raise ServerError(str(e)) from e


class LlamaServer:
    """One ``llama-server`` process serving one model.

    Args:
        options: How to start the server.
        model: The GGUF file.
        log_path: Where the server output is appended.
    """

    def __init__(self, options: ServerOptions, model: Path, log_path: Path) -> None:
        self.options = options
        self.model = model
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.started_at = 0.0
        """Time of the last successful start."""

    @property
    def base(self) -> str:
        return f"http://{self.options.host}:{self.options.port}"

    def _binary(self) -> str:
        return shutil.which(self.options.binary) or self.options.binary

    def _env(self) -> dict:
        # llama.cpp release builds keep their shared libraries next to the executable
        env = dict(os.environ)
        lib = str(Path(self._binary()).parent)
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        return env

    def devices(self) -> list[str]:
        """Compute devices the server sees (``--list-devices``), for example
        ``"Vulkan0: Radeon 8060S Graphics (...)"``; empty if it sees only the CPU.
        """
        try:
            out = subprocess.run(
                [self._binary(), "--list-devices"], capture_output=True, text=True, timeout=120, env=self._env()
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ServerError(f"cannot run llama-server --list-devices: {e}") from e
        lines = [ln.strip() for ln in (out.stdout + out.stderr).splitlines()]
        if "Available devices:" not in lines:
            return []
        start = lines.index("Available devices:") + 1
        return [ln for ln in lines[start:] if ln and ln != "(none)"]

    def start(self, ready_timeout_s: float = 900) -> None:
        """Start the server and wait until ``/health`` answers.

        Raises:
            ServerError: If no GPU is visible while one is required, or the server stops or is not
                ready in time.
        """
        if self.options.require_gpu and not self.devices():
            raise ServerError(
                "llama-server sees no GPU: the model would run on the CPU. Check that the "
                "user can access the GPU (on Linux, membership of the 'render' group)"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        command = [self._binary(), *self.options.command(self.model)[1:]]
        self.proc = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=self._env(), start_new_session=True
        )
        deadline = time.time() + ready_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise ServerError(f"llama-server exited at start (code {self.proc.returncode}), see {self.log_path}")
            try:
                if _request(self.base + "/health", None, 5).get("status") == "ok":
                    self.started_at = time.time()
                    return
            except (ServerError, RequestTimeout):
                pass
            time.sleep(1)
        self.stop()
        raise ServerError(f"llama-server not ready after {ready_timeout_s:.0f} s")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Stop the server. Safe to call from several threads at once (shutdown and normal end)."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        except ProcessLookupError:
            pass  # already gone

    def restart(self) -> None:
        self.stop()
        self.start()

    def props(self) -> dict:
        """Server properties (build, context, chat template), recorded with the run."""
        return _request(self.base + "/props", None, 30)

    def count_tokens(self, text: str) -> int:
        return len(_request(self.base + "/tokenize", {"content": text}, 60)["tokens"])

    def chat(self, payload: dict, timeout_s: float) -> dict:
        """Send an OpenAI-style chat completion request."""
        if not self.alive():
            raise ServerError("llama-server is not running")
        return _request(self.base + "/v1/chat/completions", payload, timeout_s)

    def systemone(self, payload: dict, timeout_s: float) -> dict:
        """Send a decision request to ``/v1/systemone``: the server builds the prompt with the model's
        template and returns the chosen option with the probability of each option.
        """
        if not self.alive():
            raise ServerError("llama-server is not running")
        return _request(self.base + "/v1/systemone", payload, timeout_s)
