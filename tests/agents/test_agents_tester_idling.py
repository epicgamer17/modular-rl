import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

import copy
import torch
import time
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.workers.tester import Tester, StandardGymTest
from agents.action_selectors.selectors import ArgmaxSelector
from tests.agents.conftest import MockQValueNetwork as MockNetwork, MockGymEnv


def make_mock_env():
    return MockGymEnv()


def _setup_tester_idling_context(rainbow_cartpole_replay_config, make_cartpole_config):
    device = torch.device("cpu")
    network = MockNetwork(num_actions=2).to(device)
    import torch.multiprocessing as mp

    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    selector = ArgmaxSelector()

    game_config = make_cartpole_config(
        max_score=1.0,
        min_score=0.0,
        is_discrete=True,
        is_image=False,
        is_deterministic=True,
        has_legal_moves=True,
        perfect_information=True,
        multi_agent=False,
        num_players=1,
        num_actions=2,
        env_factory=make_mock_env,
    )
    config = copy.deepcopy(rainbow_cartpole_replay_config)
    config.game = game_config
    config.test_trials = 1
    return device, network, selector, config


def test_tester_execution_throttling(
    rainbow_cartpole_replay_config, make_cartpole_config
):
    """
    Verify that Tester only executes when request_work is called.
    """
    device, network, selector, config = _setup_tester_idling_context(
        rainbow_cartpole_replay_config, make_cartpole_config
    )
    executor = TorchMPExecutor()
    test_type = StandardGymTest("std", num_trials=1)

    launch_args = (
        make_mock_env,
        network,
        selector,
        None,
        1,
        config,
        device,
        "tester",
        [test_type],
    )

    try:
        # 1. Launch Tester
        try:
            executor.launch(Tester, launch_args, num_workers=1)
        except RuntimeError as err:
            if "Operation not permitted" in str(err) or "torch_shm_manager" in str(err):
                pytest.skip(
                    "Shared-memory multiprocessing is unavailable in this test environment"
                )
            raise

        # Wait briefly to ensure it doesn't immediately dump results
        time.sleep(0.5)

        # 2. Check result queue - should be empty since we haven't triggered it
        results, _ = executor.collect_data(min_samples=None, worker_type=Tester)
        assert len(results) == 0, "Tester should not return results until triggered."

        # 3. Request work
        executor.request_work(Tester)

        # 4. Wait for exactly 1 result (might take a moment to process)
        start_time = time.time()
        results = []
        while len(results) < 1 and time.time() - start_time < 2.0:
            new_results, _ = executor.collect_data(min_samples=None, worker_type=Tester)
            results.extend(new_results)
            time.sleep(0.1)

        assert (
            len(results) == 1
        ), "Tester should return exactly 1 result after being triggered."
        assert "std" in results[0]

        # 5. Check result queue again - should be empty until another trigger
        time.sleep(0.5)
        no_results, _ = executor.collect_data(min_samples=None, worker_type=Tester)
        assert (
            len(no_results) == 0
        ), "Tester should not run continuously without triggers."

        # 6. Request work again to ensure repeatability
        executor.request_work(Tester)
        start_time = time.time()
        results_2 = []
        while len(results_2) < 1 and time.time() - start_time < 2.0:
            new_results, _ = executor.collect_data(min_samples=None, worker_type=Tester)
            results_2.extend(new_results)
            time.sleep(0.1)

        assert (
            len(results_2) == 1
        ), "Tester should return another result after a second trigger."

    finally:
        executor.stop()
