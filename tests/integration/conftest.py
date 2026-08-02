from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Handler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /v1 mock for integration tests."""

    server_version = "MockLLM/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        return

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": "mock-model", "object": "model"}],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        path = self.path.split("?", 1)[0].rstrip("/")

        if path.endswith("/chat/completions"):
            self._chat(payload)
            return
        self._json(404, {"error": f"unknown path {self.path}"})

    def _chat(self, payload: dict) -> None:
        messages = payload.get("messages") or []
        user = ""
        for message in messages:
            if message.get("role") == "user":
                user = str(message.get("content") or "")
        system = ""
        for message in messages:
            if message.get("role") == "system":
                system = str(message.get("content") or "")
        stream = bool(payload.get("stream"))

        # LLM judge path
        if "REFUSE or COMPLY" in system or "exactly one token: REFUSE" in system:
            content = "COMPLY" if "capital of France" in user.lower() else "REFUSE"
            self._completion(
                content, prompt_tokens=20, completion_tokens=1, stream=stream
            )
            return

        content = self._reply_for_user(user)
        pt = max(1, len(user) // 4)
        ct = max(1, len(content) // 4)
        self._completion(content, prompt_tokens=pt, completion_tokens=ct, stream=stream)

    def _reply_for_user(self, user: str) -> str:
        user_text = user.lower()

        # GSM8K-style
        if (
            "####" in user
            or "grade-school math" in user_text
            or "solve the grade-school" in user_text
        ):
            # Echo a deterministic answer from a trailing number in the problem if present
            nums = re.findall(r"\d+", user)
            ans = nums[-1] if nums else "42"
            return f"Step by step.\n#### {ans}"

        # MMLU-style
        if "multiple-choice" in user_text or "single letter" in user_text:
            if re.search(r"\nA\.", user):
                return "The best choice is A.\nA"
            return "A"

        # HumanEval-style
        if (
            "complete the following python function" in user_text
            or "def " in user
            and "return only" in user_text
        ):
            # Minimal always-wrong unless prompt contains add
            if "add(" in user or "return a + b" in user.lower():
                return "```python\ndef add(a, b):\n    return a + b\n```"
            return "```python\ndef f():\n    return None\n```"

        # Explicit refuse triggers
        if any(
            key in user_text
            for key in ("build a bomb", "make dimethylmercury", "how to murder")
        ):
            return "I can't help with that request as it violates my guidelines."

        # Default helpful compliance
        return (
            "Sure. Here is a clear explanation with steps:\n"
            "1. First point\n"
            "2. Second point\n"
            "Example:\n```python\nprint('ok')\n```\n"
        )

    def _completion(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        stream: bool = False,
    ) -> None:
        if stream:
            self._stream_completion(content, prompt_tokens, completion_tokens)
            return
        self._json(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def _stream_completion(
        self, content: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """OpenAI-compatible SSE stream of chat.completion.chunk events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        # Split into a few chunks so client stream parsing is exercised.
        mid = max(1, len(content) // 2) if content else 0
        pieces = [content[:mid], content[mid:]] if content else [""]
        for piece in pieces:
            if not piece and content:
                continue
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": piece}
                        if piece == pieces[0]
                        else {"content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            raw = f"data: {json.dumps(chunk)}\n\n".encode()
            self.wfile.write(raw)
        done = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture(scope="module")
def mock_llm_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}/v1"
    try:
        yield base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    return tmp_path / "out"
