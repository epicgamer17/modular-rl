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
        def __init__(self):
            self.enabled = False
            self.mode = "default"
            self.fullgraph = False

    def __init__(self):
        self.compilation = MockConfig.Compilation()


class SlowActor:
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
        self.worker_id = worker_id

    def setup(self):
        pass

    def update_parameters(self, params):
        pass

    def play_sequence(self):
        time.sleep(0.05)
        return {"worker_id": self.worker_id, "episode_length": 1}


def _collect_some_results(executor: TorchMPExecutor, timeout_s: float = 5.0) -> int:
    total = 0
    deadline = time.time() + timeout_s
    while time.time() < deadline and total < 2:
        batch, _ = executor.collect_data(min_samples=None, worker_type=SlowActor)
        total += len(batch)
        time.sleep(0.05)
    return total


@pytest.mark.regression
def test_regression_slot_leak():
    """
    Regression for executor resource leaks.
    Ensures repeated launch/collect/stop cycles still produce data.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    config = MockConfig()
    worker_args = (
        _env_factory,
        None,
        None,
        None,
        1,
        config,  # TorchMPExecutor expects config at index 5.
        torch.device("cpu"),
        "regression_slot_leak",
    )

    executor = TorchMPExecutor()
    try:
        executor.launch(SlowActor, worker_args, num_workers=2)
        collected_first = _collect_some_results(executor)
    finally:
        executor.stop()

    assert collected_first > 0
    assert executor.workers == []

    # Re-run a second cycle to verify no residual resource leak blocks relaunch.
    executor2 = TorchMPExecutor()
    try:
        executor2.launch(SlowActor, worker_args, num_workers=2)
        collected_second = _collect_some_results(executor2)
    finally:
        executor2.stop()

    assert collected_second > 0
    assert executor2.workers == []
