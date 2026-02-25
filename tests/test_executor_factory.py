import os
import sys
import torch
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.executors.factory import create_executor
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.executors.puffer_executor import PufferExecutor


def test_executor_factory():
    print("Testing Executor Factory...")

    config = MagicMock()
    env_factory = MagicMock()
    agent_network = MagicMock()
    replay_buffer = MagicMock()
    action_selector = MagicMock()
    search_alg = MagicMock()

    # Test TorchMP routing
    config.executor_type = "torch_mp"
    executor = create_executor(config=config)
    assert isinstance(
        executor, TorchMPExecutor
    ), f"Expected TorchMPExecutor, got {type(executor)}"
    print("✓ TorchMP routing verified.")

    # Test Puffer routing
    config.executor_type = "puffer"
    executor = create_executor(config=config)
    assert isinstance(
        executor, PufferExecutor
    ), f"Expected PufferExecutor, got {type(executor)}"
    assert executor.config == config
    print("✓ Puffer routing verified.")

    # Test default routing (should be torch_mp)
    del config.executor_type
    executor = create_executor(config=config)
    assert isinstance(executor, TorchMPExecutor), "Default should be torch_mp"
    print("✓ Default routing verified.")

    print("\nAll executor factory tests passed!")


if __name__ == "__main__":
    test_executor_factory()
