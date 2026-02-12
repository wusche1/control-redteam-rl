import asyncio
import json
from pathlib import Path

import tinker
from tinker_cookbook import model_info

from .train_grpo import ControlEvalDatasetBuilder, make_winrate_evaluator


async def eval_only(
    model_name: str,
    log_path: str,
    task_type: str,
    task_config: dict,
    batch_size: int,
    group_size: int,
    eval_group_size: int | None,
    max_tokens: int,
    max_turns: int,
    code_timeout: float,
    prompts: dict,
    monitor_model: str,
    monitor_prompt: str,
    main_task_reward: float,
    side_task_reward: float,
    monitor_penalty: float,
    max_test_samples: int | None,
    eval_log_completions: int,
    adapter_path: str | None,
    renderer_name: str | None,
    **_,
):
    if not renderer_name:
        renderer_name = model_info.get_recommended_renderer_name(model_name)

    dataset_builder = ControlEvalDatasetBuilder(
        task_type=task_type,
        task_config=task_config,
        batch_size=batch_size,
        group_size=group_size,
        eval_group_size=eval_group_size,
        model_name=model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        log_completions_per_step=-1,
        eval_log_completions=eval_log_completions,
        prompts=prompts,
        monitor_model=monitor_model,
        monitor_prompt=monitor_prompt,
        main_task_reward=main_task_reward,
        side_task_reward=side_task_reward,
        monitor_penalty=monitor_penalty,
        max_tokens=max_tokens,
        max_turns=max_turns,
        code_timeout=code_timeout,
        max_train_samples=0,
        max_test_samples=max_test_samples,
        start_batch=0,
    )

    if adapter_path:
        sampling_client = tinker.ServiceClient().create_sampling_client(model_path=adapter_path)
    else:
        sampling_client = tinker.ServiceClient().create_sampling_client(base_model=model_name)

    await dataset_builder()
    evaluator = make_winrate_evaluator(dataset_builder, max_tokens)
    metrics = await evaluator(sampling_client)

    Path(log_path).mkdir(parents=True, exist_ok=True)
    (Path(log_path) / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def main(**kwargs):
    asyncio.run(eval_only(**kwargs))
