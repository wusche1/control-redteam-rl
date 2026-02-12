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
    match = re.search(r'\{[^{}]*"suspicion_score"[^{}]*\}', content)
    if match:
        return match.group(0)
    match = re.search(r'\{[^{}]*"suspiciousness"[^{}]*\}', content)
    if match:
        return match.group(0)
    match = re.search(r'\{[^{}]+\}', content)
    if match:
        return match.group(0)
    return content


async def _monitor_async(
    model: str,
    prompt_template: str,
    task: str = "",
    actions: str = "",
    output: str = "",
) -> dict:
    prompt = prompt_template.replace("{task}", task).replace("{actions}", actions).replace("{output}", output)
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
        # Normalize key name: accept both "suspicion_score" and "suspiciousness"
        if "suspicion_score" in result:
            result["suspicion_score"] = float(result["suspicion_score"])
        elif "suspiciousness" in result:
            result["suspicion_score"] = float(result["suspiciousness"])
        else:
            print(f"[monitor] Missing suspicion score in: {raw_content}")
            result["suspicion_score"] = 0.5
        result["suspicion_score"] = max(0.0, min(1.0, result["suspicion_score"]))
        return result
    except Exception as e:
        print(f"[monitor] Error: {e}")
        return {"suspicion_score": 0.5, "reasoning": None}
