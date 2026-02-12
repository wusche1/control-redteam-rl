from ..base import ControlTask
from .tasks import get_tasks


def make_task(
    task_ids: list[str] | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
) -> ControlTask:
    all_tasks = get_tasks(task_ids)
    # For bash_arena, train and test use the same task definitions
    # (the RL agent learns to solve them better over time)
    train_data = all_tasks[:max_train] if max_train else all_tasks
    test_data = all_tasks[:max_test] if max_test else all_tasks
    return ControlTask(
        name="bash_arena",
        train_data=train_data,
        test_data=test_data,
        task_type="bash_arena",
    )
