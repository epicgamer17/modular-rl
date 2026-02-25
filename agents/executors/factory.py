from configs.base import Config
from typing import Any


def create_executor(config: Config) -> Any:
    """
    Factory creating the appropriate executor based on config.
    """
    executor_type = config.executor_type.lower()

    # Applying the corrected structure:
    if executor_type == "puffer":
        from .puffer_executor import PufferExecutor

        return PufferExecutor(config)
    elif executor_type == "local":
        from .local_executor import LocalExecutor

        return LocalExecutor()
    elif executor_type == "torch_mp":  # Corrected from 'else' to 'elif'
        from .torch_mp_executor import TorchMPExecutor

        return TorchMPExecutor()
    else:
        raise ValueError(f"Unknown executor_type: {executor_type}")
