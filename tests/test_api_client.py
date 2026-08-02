from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from harness.api_client import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ChatClient,
    ChatResult,
    estimate_tokens,
    parse_sse_chat_stream,
    parse_usage,
    resolve_base_url,
)


def test_resolve_base_url_defaults_local_ollama():
    assert resolve_base_url() == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1"
    assert DEFAULT_PORT == 11434
    assert DEFAULT_HOST == "127.0.0.1"


def test_resolve_base_url_host_port_and_override():
    assert resolve_base_url(host="10.0.0.5", port=2000) == "http://10.0.0.5:2000/v1"
    assert (
        resolve_base_url("http://example.com:8080/v1", host="ignored", port=1)
        == "http://example.com:8080/v1"
    )
    with pytest.raises(ValueError):
        resolve_base_url(port=0)


def test_estimate_and_parse_usage():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    prompt_tokens, completion_tokens, total_tokens, estimated = parse_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
        "p",
        "c",
    )
    assert (prompt_tokens, completion_tokens, total_tokens, estimated) == (
        10,
        20,
        30,
        False,
    )
    prompt_tokens, completion_tokens, total_tokens, estimated = parse_usage(
        {}, "hello world!!", "resp text here"
    )
    assert estimated is True
    assert (
        prompt_tokens > 0
        and completion_tokens > 0
        and total_tokens == prompt_tokens + completion_tokens
    )


def test_parse_sse_chat_stream_accumulates_content_and_usage():
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
        "data: [DONE]",
    ]
    content, reasoning, usage = parse_sse_chat_stream(lines)
    assert content == "hello"
    assert reasoning == ""
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 2


def test_parse_sse_captures_reasoning_separately():
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"think..."}}]}',
        'data: {"choices":[{"delta":{"content":"answer"}}]}',
        "data: [DONE]",
    ]
    content, reasoning, _ = parse_sse_chat_stream(lines)
    assert content == "answer"
    assert reasoning == "think..."


def test_base_url_required():
    with patch("harness.api_client.httpx.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        with pytest.raises(ValueError, match="base_url is required"):
            ChatClient(base_url="", model="m")


def test_base_url_rstrip_slash():
    with patch("harness.api_client.httpx.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        client = ChatClient(base_url="http://example.com/v1/", model="m")
        assert client.base_url == "http://example.com/v1"


class _StreamCM:
    """Context manager mimicking httpx stream response."""

    def __init__(self, response: MagicMock):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, *args):
        return False


def _stream_response(
    status_code: int,
    lines: list[str] | None = None,
    *,
    raise_on_status: bool = True,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if status_code >= 400 and raise_on_status:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status = MagicMock()
    resp.read = MagicMock()
    resp.iter_lines.return_value = iter(lines or [])
    return resp


def test_chat_success_streams_chat_result():
    lines = [
        'data: {"choices":[{"delta":{"content":"hello world"}}]}',
        'data: {"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}',
        "data: [DONE]",
    ]
    mock_http = MagicMock()
    mock_http.stream.return_value = _StreamCM(_stream_response(200, lines))

    with patch("harness.api_client.httpx.Client", return_value=mock_http):
        client = ChatClient(base_url="http://x/v1", model="mod-a", api_key="k")
        out = client.chat("hi", system="sys", temperature=0.2, max_tokens=64)
        assert isinstance(out, ChatResult)
        assert out.content == "hello world"
        assert out.prompt_tokens == 5
        assert out.completion_tokens == 10
        assert out.total_tokens == 15
        assert out.tokens_estimated is False
        assert out.latency_s >= 0
        body = mock_http.stream.call_args.kwargs["json"]
        assert body["model"] == "mod-a"
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        client.close()
        mock_http.close.assert_called_once()


def test_chat_reasoning_fallback_when_content_empty():
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"long think"}}]}',
        'data: {"usage":{"prompt_tokens":1,"completion_tokens":50,"total_tokens":51}}',
        "data: [DONE]",
    ]
    mock_http = MagicMock()
    mock_http.stream.return_value = _StreamCM(_stream_response(200, lines))

    with patch("harness.api_client.httpx.Client", return_value=mock_http):
        client = ChatClient(base_url="http://x/v1", model="m")
        out = client.chat("q")
        assert out.content == "long think"
        assert out.reasoning_content == "long think"
        assert out.used_reasoning_fallback is True


def test_chat_model_override():
    lines = [
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        "data: [DONE]",
    ]
    mock_http = MagicMock()
    mock_http.stream.return_value = _StreamCM(_stream_response(200, lines))

    with patch("harness.api_client.httpx.Client", return_value=mock_http):
        client = ChatClient(base_url="http://x/v1", model="default")
        client.chat("q", model="other")
        assert mock_http.stream.call_args.kwargs["json"]["model"] == "other"


def test_chat_empty_content_estimates_tokens():
    lines = ["data: [DONE]"]
    mock_http = MagicMock()
    mock_http.stream.return_value = _StreamCM(_stream_response(200, lines))

    with patch("harness.api_client.httpx.Client", return_value=mock_http):
        client = ChatClient(base_url="http://x/v1", model="m")
        out = client.chat("q")
        assert out.content == ""
        assert out.tokens_estimated is True


def test_ssl_wrong_version_hint():
    mock_http = MagicMock()
    mock_http.stream.side_effect = httpx.ConnectError(
        "[SSL: WRONG_VERSION_NUMBER] wrong version number"
    )

    with patch("harness.api_client.httpx.Client", return_value=mock_http):
        client = ChatClient(base_url="https://x/v1", model="m")
        with pytest.raises(ConnectionError, match="http://HOST:PORT/v1"):
            client.chat("q")


def test_other_connect_error_passthrough():
    mock_http = MagicMock()
    mock_http.stream.side_effect = httpx.ConnectError("connection refused")

    with patch("harness.api_client.httpx.Client", return_value=mock_http):
        client = ChatClient(base_url="http://x/v1", model="m")
        with pytest.raises(httpx.ConnectError, match="connection refused"):
            client.chat("q")


def test_chat_retries_502_then_succeeds():
    ok_lines = [
        'data: {"choices":[{"delta":{"content":"recovered"}}]}',
        "data: [DONE]",
    ]
    mock_http = MagicMock()
    mock_http.stream.side_effect = [
        _StreamCM(_stream_response(502)),
        _StreamCM(_stream_response(503)),
        _StreamCM(_stream_response(200, ok_lines)),
    ]

    with (
        patch("harness.api_client.httpx.Client", return_value=mock_http),
        patch("harness.api_client.time.sleep") as sleep,
    ):
        client = ChatClient(base_url="http://x/v1", model="m", retry_sleep_s=5.0)
        out = client.chat("q")
        assert out.content == "recovered"
        assert mock_http.stream.call_count == 3
        assert sleep.call_count == 2
        sleep.assert_called_with(5.0)


def test_chat_retries_exhausted_raises():
    mock_http = MagicMock()
    mock_http.stream.return_value = _StreamCM(_stream_response(503))

    with (
        patch("harness.api_client.httpx.Client", return_value=mock_http),
        patch("harness.api_client.time.sleep") as sleep,
    ):
        client = ChatClient(
            base_url="http://x/v1", model="m", max_retries=2, retry_sleep_s=5.0
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.chat("q")
        assert mock_http.stream.call_count == 3
        assert sleep.call_count == 2


def test_chat_does_not_retry_other_http_errors():
    mock_http = MagicMock()
    mock_http.stream.return_value = _StreamCM(_stream_response(400))

    with (
        patch("harness.api_client.httpx.Client", return_value=mock_http),
        patch("harness.api_client.time.sleep") as sleep,
    ):
        client = ChatClient(base_url="http://x/v1", model="m")
        with pytest.raises(httpx.HTTPStatusError):
            client.chat("q")
        assert mock_http.stream.call_count == 1
        sleep.assert_not_called()
