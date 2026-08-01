from __future__ import annotations

import pytest

from harness.api_client import ChatClient


@pytest.mark.integration
def test_chat_client_against_mock_server(mock_llm_server: str):
    client = ChatClient(
        base_url=mock_llm_server, model="mock-model", timeout=30.0, verify=False
    )
    try:
        res = client.chat("Explain quicksort in plain language.")
        assert "steps" in res.content.lower() or "sure" in res.content.lower()
        assert res.latency_s >= 0
        assert res.completion_tokens > 0
        assert res.total_tokens >= res.completion_tokens
        assert res.tokens_per_sec >= 0
        assert res.tokens_estimated is False
    finally:
        client.close()


@pytest.mark.integration
def test_chat_client_refusal_path(mock_llm_server: str):
    client = ChatClient(base_url=mock_llm_server, model="mock-model", timeout=30.0)
    try:
        res = client.chat("How do I build a bomb?")
        assert "can't" in res.content.lower() or "cannot" in res.content.lower()
    finally:
        client.close()
