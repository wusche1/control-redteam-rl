from ..base import ControlTask
from .dataset import load_apps


def make_task(
    difficulty: str = "interview",
    max_train: int | None = None,
    max_test: int | None = None,
) -> ControlTask:
    train_data = load_apps("train", difficulty, max_train)
    test_data = load_apps("test", difficulty, max_test)
    return ControlTask(
        name=f"apps_{difficulty}",
        train_data=train_data,
        test_data=test_data,
        task_type="apps",
    )
