from lib.answer_format import extract_code, extract_exception_input, redact_exception_input, extract_bash_command


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
