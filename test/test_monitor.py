import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from lib.monitor import _monitor_async, _extract_json


class TestExtractJson:
    def test_plain_json(self):
        result = json.loads(_extract_json('{"suspicion_score": 0.3, "reasoning": "ok"}'))
        assert result["suspicion_score"] == 0.3

    def test_markdown_wrapped(self):
        text = '```json\n{"suspicion_score": 0.7, "reasoning": "bad"}\n```'
        result = json.loads(_extract_json(text))
        assert result["suspicion_score"] == 0.7

    def test_preamble_text(self):
        text = 'Here is my analysis:\n{"suspicion_score": 0.2, "reasoning": "clean"}'
        result = json.loads(_extract_json(text))
        assert result["suspicion_score"] == 0.2


class TestMonitorAsync:
    def test_successful_call(self):
        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.usage.cost = 0.001
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"suspicion_score": 3, "reasoning": "looks ok"}'))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("lib.monitor._get_client", return_value=mock_client):
            result = asyncio.run(_monitor_async("model", "Review {task}: {actions}", task="task", actions="code"))

        assert result["suspicion_score"] == 0.3  # 3/10
        assert result["reasoning"] == "looks ok"

    def test_legacy_suspiciousness_key(self):
        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.usage.cost = 0.001
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"suspiciousness": 7, "reasoning": "suspicious"}'))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("lib.monitor._get_client", return_value=mock_client):
            result = asyncio.run(_monitor_async("model", "Review {task}: {actions}", task="task", actions="code"))

        assert result["suspicion_score"] == 0.7  # 7/10

    def test_score_key(self):
        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.usage.cost = 0.001
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"score": 5, "reasoning": "moderate"}'))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("lib.monitor._get_client", return_value=mock_client):
            result = asyncio.run(_monitor_async("model", "Review {task}: {actions}", task="task", actions="code"))

        assert result["suspicion_score"] == 0.5  # 5/10

    def test_placeholder_substitution(self):
        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.usage.cost = 0.001
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"suspicion_score": 1, "reasoning": "clean"}'))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("lib.monitor._get_client", return_value=mock_client):
            result = asyncio.run(_monitor_async(
                "model", "Task: {task}\nActions: {actions}\nFull: {output}",
                task="solve X", actions="print(1)", output="I'll solve X\nprint(1)",
            ))

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Task: solve X" in prompt
        assert "Actions: print(1)" in prompt
        assert "Full: I'll solve X" in prompt
