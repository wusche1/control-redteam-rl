import re
import json


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_tool_call(message: dict, tool_name: str) -> dict | None:
    """Extract arguments for a specific tool call from a parsed message.

    Works with tinker_cookbook's renderer.parse_response() output, which puts
    ToolCall objects in message["tool_calls"].
    """
    for tc in message.get("tool_calls", []):
        if tc.function.name == tool_name:
            return json.loads(tc.function.arguments)
    return None


def extract_tool_calls(message: dict, tool_name: str) -> list[dict]:
    """Extract all tool calls for a given tool name from a parsed message."""
    results = []
    for tc in message.get("tool_calls", []):
        if tc.function.name == tool_name:
            results.append(json.loads(tc.function.arguments))
    return results


def decompose_message(message: dict) -> dict:
    """Split a parsed message into reasoning and text parts.

    Handles both plain string content and list-of-ContentPart content
    (ThinkingPart/TextPart TypedDicts from tinker_cookbook renderers).
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return {"reasoning": "", "text": content}
    reasoning_parts = []
    text_parts = []
    for part in content:
        if part.get("type") == "thinking":
            reasoning_parts.append(part.get("thinking", ""))
        elif part.get("type") == "text":
            text_parts.append(part.get("text", ""))
    return {
        "reasoning": "\n".join(reasoning_parts),
        "text": "\n".join(text_parts),
    }


def format_tool_response(tool_call_id: str | None, name: str, content: str) -> dict:
    """Create a tool response message compatible with tinker_cookbook renderers."""
    msg = {"role": "tool", "content": content, "name": name}
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


# Legacy text-based extraction (kept for backward compatibility / tests)

def extract_code(text: str) -> str | None:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_exception_input(text: str) -> str | None:
    match = re.search(r"<EXCEPTION_INPUT>\s*\n?(.*?)\s*</EXCEPTION_INPUT>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def redact_exception_input(text: str) -> str:
    return re.sub(r"\s*<EXCEPTION_INPUT>.*?</EXCEPTION_INPUT>\s*", "", text, flags=re.DOTALL).strip()


def extract_bash_command(text: str) -> str | None:
    match = re.search(r"<bash>\s*(.*?)\s*</bash>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_all_bash_commands(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"<bash>\s*(.*?)\s*</bash>", text, re.DOTALL)]
