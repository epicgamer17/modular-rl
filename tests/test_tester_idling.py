import unittest
import torch
import time
from configs.base import Config
from configs.games.game import GameConfig
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.workers.tester import Tester, StandardGymTest
from agents.action_selectors.selectors import ArgmaxSelector

import types


class MockNetwork(torch.nn.Module):
    def __init__(self, num_actions=2):
        super().__init__()
        self.num_actions = num_actions
        self.param = torch.nn.Parameter(torch.zeros(1))

    def obs_inference(self, obs: torch.Tensor):
        class Output:
            def __init__(self, num_actions, batch_size):
                self.q_values = torch.zeros((batch_size, num_actions))
                self.q_values[:, 1] = 1.0  # action 1 is better

        return Output(self.num_actions, obs.shape[0])


class MockEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 3

    def reset(self, **kwargs):
        self.step_count = 0
        return [0.0], {"legal_moves": [[0, 1]]}

    def step(self, action):
        self.step_count += 1
        return (
            [0.0],
            1.0,
            self.step_count >= self.max_steps,
            False,
            {"legal_moves": [[0, 1]]},
        )

    def close(self):
        pass


def make_mock_env():
    return MockEnv()


class TestTesterIdling(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.network = MockNetwork(num_actions=2).to(self.device)
        import torch.multiprocessing as mp

        try:
            mp.set_sharing_strategy("file_system")
        except Exception:
            pass
        self.selector = ArgmaxSelector()

        game_config = GameConfig(
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
            make_env=make_mock_env,
        )
        self.config = Config(config_dict={"test_trials": 1}, game_config=game_config)

    def test_tester_execution_throttling(self):
        """
        Verify that Tester only executes when request_work is called.
        """
        executor = TorchMPExecutor()
        test_type = StandardGymTest("std", num_trials=1)

        launch_args = (
            make_mock_env,
            self.network,
            self.selector,
            1,
            self.config,
            self.device,
            "tester",
            [test_type],
        )

        try:
            # 1. Launch Tester
            executor.launch(Tester, launch_args, num_workers=1)

            # Wait briefly to ensure it doesn't immediately dump results
            time.sleep(0.5)

            # 2. Check result queue - should be empty since we haven't triggered it
            results, _ = executor.collect_data(min_samples=None, worker_type=Tester)
            self.assertEqual(
                len(results), 0, "Tester should not return results until triggered."
            )

            # 3. Request work
            executor.request_work(Tester)

            # 4. Wait for exactly 1 result (might take a moment to process)
            start_time = time.time()
            results = []
            while len(results) < 1 and time.time() - start_time < 2.0:
                new_results, _ = executor.collect_data(
                    min_samples=None, worker_type=Tester
                )
                results.extend(new_results)
                time.sleep(0.1)

            self.assertEqual(
                len(results),
                1,
                "Tester should return exactly 1 result after being triggered.",
            )
            self.assertIn("std", results[0])

            # 5. Check result queue again - should be empty until another trigger
            time.sleep(0.5)
            no_results, _ = executor.collect_data(min_samples=None, worker_type=Tester)
            self.assertEqual(
                len(no_results),
                0,
                "Tester should not run continuously without triggers.",
            )

            # 6. Request work again to ensure repeatability
            executor.request_work(Tester)
            start_time = time.time()
            results_2 = []
            while len(results_2) < 1 and time.time() - start_time < 2.0:
                new_results, _ = executor.collect_data(
                    min_samples=None, worker_type=Tester
                )
                results_2.extend(new_results)
                time.sleep(0.1)

            self.assertEqual(
                len(results_2),
                1,
                "Tester should return another result after a second trigger.",
            )

        finally:
            executor.stop()


if __name__ == "__main__":
    unittest.main()
