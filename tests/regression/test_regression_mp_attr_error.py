import time
from typing import Any, Callable, Optional

import pytest
import torch
import torch.multiprocessing as mp

from agents.executors.torch_mp_executor import TorchMPExecutor


def _env_factory():
    return None


class MockConfig:
    class Compilation:
        def __init__(self, enabled):
            self.enabled = enabled
            self.mode = "default"
            self.fullgraph = False

    def __init__(self, enabled=False):
        self.compilation = MockConfig.Compilation(enabled)


class MockActor:
    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_network: Any,
        action_selector: Any,
        replay_buffer: Any,
        num_players: int,
        config: Any,
        device: Optional[torch.device] = None,
        name: str = "agent",
        *,
        worker_id: int = 0,
    ):
        self.config = config
        self.worker_id = worker_id

    def setup(self):
        pass

    def update_parameters(self, params):
        pass

    def play_sequence(self):
        time.sleep(0.05)
        return {"done": True, "episode_length": 1}


@pytest.mark.regression
def test_regression_mp_attr_error():
    """
    Regression for multiprocessing worker launch path that previously crashed
    due to misplaced config access/attribute assumptions.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    executor = TorchMPExecutor()
    config = MockConfig(enabled=True)

    args = (
        _env_factory,
        None,
        None,
        None,
        1,
        config,  # TorchMPExecutor expects config at index 5.
        torch.device("cpu"),
        "test_agent",
    )

    try:
        executor.launch(MockActor, args, num_workers=2)

        results = []
        deadline = time.time() + 5.0
        while time.time() < deadline and not results:
            batch, _ = executor.collect_data(min_samples=None, worker_type=MockActor)
            results.extend(batch)
            time.sleep(0.05)

        assert results, "No results collected from multiprocessing workers."
        assert all(r.get("done") is True for r in results)
    finally:
        executor.stop()
