from dataclasses import dataclass


@dataclass
class ControlTask:
    name: str
    train_data: list[dict]
    test_data: list[dict]
    task_type: str  # "apps" or "bash_arena"
