import torch
import torch.multiprocessing as mp
import time
from typing import Tuple, Any, Callable, Optional
from agents.executors.torch_mp_executor import TorchMPExecutor


class MockConfig:
    def __init__(self, enabled=False):
        class Compilation:
            def __init__(self, enabled):
                self.enabled = enabled
                self.mode = "default"
                self.fullgraph = False

        self.compilation = Compilation(enabled)


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
        # This matches the signature where config is at index 5
        self.config = config
        self.worker_id = worker_id

    def setup(self):
        pass

    def play_sequence(self):
        time.sleep(0.1)
        return {"done": True}


def test_mp_executor_staggering():
    mp.set_start_method("spawn", force=True)
    executor = TorchMPExecutor()

    config = MockConfig(enabled=True)

    # args matches the 8 positional args + worker_id kwarg pattern
    # config is at index 5
    args = (
        lambda: None,  # env_factory
        None,  # agent_network
        None,  # action_selector
        None,  # replay_buffer
        1,  # num_players
        config,  # config (INDEX 5)
        torch.device("cpu"),  # device
        "test_agent",  # name
    )

    print("Launching workers...")
    executor.launch(MockActor, args, num_workers=2)

    print("Collecting data...")
    try:
        data, stats = executor.collect_data(min_samples=1, worker_type=MockActor)
        print(f"Success! Collected {len(data)} results.")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback

        traceback.print_exc()
        executor.stop()
        raise e

    executor.stop()


if __name__ == "__main__":
    test_mp_executor_staggering()
