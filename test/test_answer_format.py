from unittest.mock import MagicMock
from lib.answer_format import (
    extract_code, extract_exception_input, redact_exception_input,
    extract_bash_command, extract_tool_call, extract_tool_calls,
    format_tool_response, decompose_message,
)


class TestExtractCode:
    def test_python_block(self):
        text = "Some text\n```python\ndef solve():\n    return 42\n```\nMore text"
        assert extract_code(text) == "def solve():\n    return 42"

    def test_plain_block(self):
        text = "```\nprint('hello')\n```"
        assert extract_code(text) == "print('hello')"

    def test_no_block(self):
        assert extract_code("no code here") is None


class TestExceptionInput:
    def test_extract(self):
        text = "code here\n<EXCEPTION_INPUT>\n5\n3 1 4\n</EXCEPTION_INPUT>"
        assert extract_exception_input(text) == "5\n3 1 4"

    def test_extract_missing(self):
        assert extract_exception_input("no tag here") is None

    def test_redact(self):
        text = "code here\n<EXCEPTION_INPUT>\n5\n</EXCEPTION_INPUT>\nafter"
        result = redact_exception_input(text)
        assert "<EXCEPTION_INPUT>" not in result
        assert "code here" in result
        assert "after" in result

    def test_roundtrip(self):
        text = "solution\n```python\nprint(1)\n```\n\n<EXCEPTION_INPUT>\n42\n</EXCEPTION_INPUT>"
        assert extract_exception_input(text) == "42"
        redacted = redact_exception_input(text)
        assert "42" not in redacted
        assert "```python" in redacted


class TestBashCommand:
    def test_extract(self):
        text = "I'll run: <bash>ls -la /etc</bash>"
        assert extract_bash_command(text) == "ls -la /etc"

    def test_multiline(self):
        text = "<bash>\napt update &&\napt install nginx\n</bash>"
        assert extract_bash_command(text) == "apt update &&\napt install nginx"

    def test_missing(self):
        assert extract_bash_command("no commands") is None


class TestExtractToolCall:
    def _make_tc(self, name, args_json):
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = args_json
        return tc

    def test_extract_submit(self):
        message = {"tool_calls": [self._make_tc("submit", '{"code": "print(1)", "secret_input": "42"}')]}
        result = extract_tool_call(message, "submit")
        assert result == {"code": "print(1)", "secret_input": "42"}

    def test_extract_bash(self):
        message = {"tool_calls": [self._make_tc("bash", '{"cmd": "ls -la"}')]}
        result = extract_tool_call(message, "bash")
        assert result == {"cmd": "ls -la"}

    def test_missing_tool(self):
        message = {"tool_calls": [self._make_tc("bash", '{"cmd": "ls"}')]}
        assert extract_tool_call(message, "submit") is None

    def test_no_tool_calls(self):
        message = {"content": "hello"}
        assert extract_tool_call(message, "bash") is None

    def test_extract_multiple(self):
        message = {"tool_calls": [
            self._make_tc("bash", '{"cmd": "ls"}'),
            self._make_tc("bash", '{"cmd": "pwd"}'),
        ]}
        results = extract_tool_calls(message, "bash")
        assert len(results) == 2
        assert results[0] == {"cmd": "ls"}
        assert results[1] == {"cmd": "pwd"}


class TestFormatToolResponse:
    def test_with_id(self):
        msg = format_tool_response("call_123", "bash", "output here")
        assert msg == {"role": "tool", "content": "output here", "name": "bash", "tool_call_id": "call_123"}

    def test_without_id(self):
        msg = format_tool_response(None, "bash", "output")
        assert msg == {"role": "tool", "content": "output", "name": "bash"}


class TestDecomposeMessage:
    def test_string_content(self):
        msg = {"role": "assistant", "content": "hello world"}
        parts = decompose_message(msg)
        assert parts["reasoning"] == ""
        assert parts["text"] == "hello world"

    def test_content_parts(self):
        msg = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Let me think..."},
            {"type": "text", "text": "The answer is 42"},
        ]}
        parts = decompose_message(msg)
        assert parts["reasoning"] == "Let me think..."
        assert parts["text"] == "The answer is 42"

    def test_no_content(self):
        msg = {"role": "assistant"}
        parts = decompose_message(msg)
        assert parts["reasoning"] == ""
        assert parts["text"] == ""

    def test_only_thinking(self):
        msg = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "reasoning here"},
        ]}
        parts = decompose_message(msg)
        assert parts["reasoning"] == "reasoning here"
        assert parts["text"] == ""
