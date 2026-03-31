from old_muzero.configs.base import Config
from typing import Any


def create_executor(config: Config) -> Any:
    """
    Factory creating the appropriate executor based on config.
    """
    executor_type = config.executor_type.lower()

    if executor_type == "local":
        from .local_executor import LocalExecutor

        return LocalExecutor()
    elif executor_type == "torch_mp":
        from .torch_mp_executor import TorchMPExecutor

        return TorchMPExecutor()
    else:
        raise ValueError(f"Unknown executor_type: {executor_type}")
