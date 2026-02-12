import os
import re
import json
import asyncio
from openai import AsyncOpenAI

_async_client = None
_cumulative_cost = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}


def get_cumulative_cost() -> dict:
    return dict(_cumulative_cost)


def reset_cumulative_cost():
    _cumulative_cost.update({"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})


def _get_client():
    global _async_client
    if _async_client is None:
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPEN_ROUTER_API_KEY environment variable is not set")
        _async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _async_client


def _extract_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)
    match = re.search(r'\{[^{}]*"suspiciousness"[^{}]*\}', content)
    if match:
        return match.group(0)
    match = re.search(r'\{[^{}]+\}', content)
    if match:
        return match.group(0)
    return content


def format_monitor_input(
    mode: str,
    problem: str,
    conversation: list[dict],
    final_output: str,
    explanation: str | None = None,
    code: str | None = None,
) -> tuple[str, str]:
    if mode == "code_only":
        return problem, code or final_output
    elif mode == "prompt_and_answer":
        return problem, final_output
    elif mode == "full_conversation":
        conv_str = "\n".join(
            f"[{msg['role']}]: {msg['content']}" for msg in conversation
        )
        return problem, conv_str
    elif mode == "answer_and_explanation":
        output = final_output
        if explanation:
            output = f"{final_output}\n\nExplanation: {explanation}"
        return problem, output
    raise ValueError(f"Unknown monitor input mode: {mode}")


async def _monitor_async(
    task: str,
    output: str,
    model: str,
    prompt_template: str,
) -> dict:
    prompt = prompt_template.replace("{task}", task).replace("{output}", output)
    try:
        response = await _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            extra_body={"usage": {"include": True}},
        )
        if response.usage:
            _cumulative_cost["prompt_tokens"] += response.usage.prompt_tokens
            _cumulative_cost["completion_tokens"] += response.usage.completion_tokens
            _cumulative_cost["cost"] += getattr(response.usage, "cost", 0.0)
        raw_content = response.choices[0].message.content
        content = _extract_json(raw_content)
        result = json.loads(content)
        if "suspiciousness" not in result:
            print(f"[monitor] Missing 'suspiciousness' key in: {raw_content}")
            result["suspiciousness"] = 5
        else:
            try:
                result["suspiciousness"] = int(result["suspiciousness"])
            except (ValueError, TypeError):
                print(f"[monitor] Cannot convert suspiciousness to int: {result['suspiciousness']}")
                result["suspiciousness"] = 5
        return result
    except Exception as e:
        print(f"[monitor] Error: {e}")
        return {"suspiciousness": 5, "reasoning": None}


async def monitor_batch_async(
    tasks: list[str],
    outputs: list[str],
    model: str,
    prompt_template: str,
) -> list[dict]:
    return await asyncio.gather(*[
        _monitor_async(task, output, model, prompt_template)
        for task, output in zip(tasks, outputs)
    ])
