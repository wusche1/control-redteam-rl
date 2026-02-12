from .base import ControlTask


def get_task(type: str, **kwargs) -> ControlTask:
    if type == "apps":
        from .apps import make_task
        return make_task(**kwargs)
    elif type == "bash_arena":
        from .bash_arena import make_task
        return make_task(**kwargs)
    raise ValueError(f"Unknown task type: {type}")
