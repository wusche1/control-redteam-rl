import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from lib.monitor import format_monitor_input, _monitor_async, _extract_json


class TestFormatMonitorInput:
    def test_prompt_and_answer(self):
        task, output = format_monitor_input(
            "prompt_and_answer", "Solve X", [], "my solution", None
        )
        assert task == "Solve X"
        assert output == "my solution"

    def test_full_conversation(self):
        convo = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Do X"},
            {"role": "assistant", "content": "Done"},
        ]
        task, output = format_monitor_input("full_conversation", "problem", convo, "Done")
        assert task == "problem"
        assert "[system]: Be helpful" in output
        assert "[assistant]: Done" in output

    def test_answer_and_explanation(self):
        task, output = format_monitor_input(
            "answer_and_explanation", "problem", [], "code here", "I did X because Y"
        )
        assert task == "problem"
        assert "code here" in output
        assert "I did X because Y" in output

    def test_answer_and_explanation_no_explanation(self):
        task, output = format_monitor_input(
            "answer_and_explanation", "problem", [], "code here", None
        )
        assert output == "code here"

    def test_code_only(self):
        task, output = format_monitor_input(
            "code_only", "Solve X", [], "Here is my solution\n```python\nprint(1)\n```", None,
            code="print(1)",
        )
        assert task == "Solve X"
        assert output == "print(1)"

    def test_code_only_fallback(self):
        task, output = format_monitor_input(
            "code_only", "Solve X", [], "no code block", None, code=None,
        )
        assert output == "no code block"

    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            format_monitor_input("invalid", "p", [], "o")


class TestExtractJson:
    def test_plain_json(self):
        result = json.loads(_extract_json('{"suspiciousness": 3, "reasoning": "ok"}'))
        assert result["suspiciousness"] == 3

    def test_markdown_wrapped(self):
        text = '```json\n{"suspiciousness": 7, "reasoning": "bad"}\n```'
        result = json.loads(_extract_json(text))
        assert result["suspiciousness"] == 7

    def test_preamble_text(self):
        text = 'Here is my analysis:\n{"suspiciousness": 2, "reasoning": "clean"}'
        result = json.loads(_extract_json(text))
        assert result["suspiciousness"] == 2


class TestMonitorAsync:
    def test_successful_call(self):
        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.usage.cost = 0.001
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"suspiciousness": 3, "reasoning": "looks ok"}'))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("lib.monitor._get_client", return_value=mock_client):
            result = asyncio.run(_monitor_async("task", "output", "model", "Review {task}: {output}"))

        assert result["suspiciousness"] == 3
        assert result["reasoning"] == "looks ok"
