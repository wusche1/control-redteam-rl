import re


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
