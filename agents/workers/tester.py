import torch
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

from modules.agent_nets.modular import ModularAgentNetwork
from agents.action_selectors.selectors import BaseActionSelector
from agents.action_selectors.types import InferenceResult
from agents.action_selectors.policy_sources import (
    BasePolicySource,
    NetworkPolicySource,
    SearchPolicySource,
)
from configs.base import Config
from replay_buffers.modular_buffer import ModularReplayBuffer


class NetworkAgent:
    """Helper agent that wraps a network and action selector for evaluations."""

    def __init__(
        self,
        name: str,
        agent_network: ModularAgentNetwork,
        action_selector: BaseActionSelector,
        device: torch.device,
    ):
        self.name = name
        self.agent_network = agent_network
        self.action_selector = action_selector
        self.device = device
        self.actor_state = None
        self.config = getattr(
            action_selector, "config", None
        )  # Fallback if config is not passed directly

        # Initialize PolicySource
        use_search = hasattr(self.config, "search_backend") and getattr(
            self.config, "search_enabled", False
        )

        if use_search:
            from search.factory import SearchBackendFactory

            search_engine = SearchBackendFactory.create(self.config)
            self.policy_source = SearchPolicySource(
                search_engine, self.agent_network, self.config
            )
        else:
            self.policy_source = NetworkPolicySource(self.agent_network)

    def predict(self, observation, info, *args, **kwargs):
        """Returns the observation unchanged, to be processed by select_actions."""
        return observation

    def select_actions(self, prediction, info, *args, **kwargs):
        """Inference using the wrapped network and selector."""
        with torch.inference_mode():
            # Ensure prediction (observation) is a tensor
            obs_tensor = torch.as_tensor(
                prediction, dtype=torch.float32, device=self.device
            )
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            # Perform inference via PolicySource
            result = self.policy_source.get_inference(
                obs=obs_tensor,
                info=info,
                agent_network=self.agent_network,
                exploration=False,
            )

            # Standardize masking
            if info is None:
                info = {}
            if "legal_moves" in info and "legal_moves_mask" not in info:
                action_tensor = result.action_dim
                assert (
                    action_tensor is not None
                ), "InferenceResult has no action tensor for mask shape"
                mask = torch.zeros(
                    action_tensor.shape, dtype=torch.bool, device=self.device
                )
                legal = info["legal_moves"]
                if isinstance(legal, (list, np.ndarray, torch.Tensor)):
                    mask[0, legal] = True
                info["legal_moves_mask"] = mask

            output = self.action_selector.select_action(
                result=result,
                info=info,
                exploration=False,
                actor_state=self.actor_state,
            )
            if isinstance(output, tuple) and len(output) == 2:
                action, self.actor_state = output
            else:
                action = output
            return action


class BaseTestType(ABC):
    """
    Abstract base class for all test types.
    Defines the interface for running a specific type of evaluation.
    """

    def __init__(self, name: str, num_trials: int):
        self.name = name
        self.num_trials = num_trials

    @abstractmethod
    def run(self, tester: "Tester", env: Any) -> Dict[str, Any]:
        """
        Executes the test using the provided tester and environment.

        Args:
            tester: The Tester instance (provides action selection).
            env: The environment to run tests in.

        Returns:
            Dictionary of metrics (score, etc.)
        """
        pass


class StandardGymTest(BaseTestType):
    """Evaluates a single-player gymnasium environment."""

    def run(self, tester: "Tester", env: Any) -> Dict[str, Any]:
        results = []
        for _ in range(self.num_trials):
            state, info = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done and episode_length < 1000:
                episode_length += 1
                action = tester.select_test_action(state, info, env)

                # Handle potential tensor/scalar action conversion
                action_val = (
                    action.item() if isinstance(action, torch.Tensor) else action
                )

                state, reward, terminated, truncated, info = env.step(action_val)
                episode_reward += float(reward)
                done = terminated or truncated

            results.append(episode_reward)

        return {
            "score": sum(results) / len(results) if results else 0.0,
            "min_score": min(results) if results else 0.0,
            "max_score": max(results) if results else 0.0,
        }


class SelfPlayTest(BaseTestType):
    """
    Evaluates a multi-player environment where all players use the agent's network.
    Typically used for zero-sum games like TicTacToe to see if it learns optimally.
    """

    def run(self, tester: "Tester", env: Any) -> Dict[str, Any]:
        # Detect players from env
        possible_agents = getattr(env, "possible_agents", [])
        num_players = len(possible_agents)
        player_results = {f"p{i}": [] for i in range(num_players)}

        for _ in range(self.num_trials):
            env.reset()
            episode_length = 0

            for agent_id in env.agent_iter():
                state, reward, terminated, truncated, info = env.last()

                if terminated or truncated:
                    break

                action = tester.select_test_action(state, info, env)

                action_val = (
                    action.item() if isinstance(action, torch.Tensor) else action
                )

                env.step(action_val)
                episode_length += 1
                if episode_length >= 1000 * num_players:
                    break

            # Record final rewards for all players
            for i, p_id in enumerate(possible_agents):
                r = float(env.rewards.get(p_id, 0.0))
                player_results[f"p{i}"].append(r)

        # Flatten for global stats
        all_rewards = [r for sublist in player_results.values() for r in sublist]
        formatted = {
            "score": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "min_score": min(all_rewards) if all_rewards else 0.0,
            "max_score": max(all_rewards) if all_rewards else 0.0,
        }
        # Add player-specific means
        for k, v in player_results.items():
            formatted[f"{k}_score"] = sum(v) / len(v) if v else 0.0

        return formatted


class VsAgentTest(BaseTestType):
    """
    Evaluates the agent against a specific fixed opponent.
    Can be used for exploitability testing or tournament play.
    """

    def __init__(self, name: str, num_trials: int, opponent: Any, player_idx: int):
        super().__init__(name, num_trials)
        self.opponent = opponent
        self.player_idx = player_idx

    def run(self, tester: "Tester", env: Any) -> Dict[str, Any]:
        results = []
        possible_agents = getattr(env, "possible_agents", [])
        num_players = len(possible_agents)

        for _ in range(self.num_trials):
            env.reset()
            episode_length = 0

            for agent_id in env.agent_iter():
                state, reward, terminated, truncated, info = env.last()

                if terminated or truncated:
                    break

                current_player_idx = possible_agents.index(agent_id)

                if current_player_idx == self.player_idx:
                    # Our agent's turn
                    action = tester.select_test_action(state, info, env)
                else:
                    # Opponent's turn
                    prediction = self.opponent.predict(state, info, env=env)
                    action = self.opponent.select_actions(prediction, info=info)

                action_val = (
                    action.item() if isinstance(action, torch.Tensor) else action
                )

                env.step(action_val)
                episode_length += 1
                if episode_length >= 1000 * num_players:
                    break

            agent_id = possible_agents[self.player_idx]
            results.append(float(env.rewards.get(agent_id, 0.0)))

        return {
            "score": sum(results) / len(results) if results else 0.0,
            "min_score": min(results) if results else 0.0,
            "max_score": max(results) if results else 0.0,
        }


class Tester:
    """
    Worker class for evaluating agents. Acts as a container for multiple test types.
    Fulfills the 'Actor' contract used by Executors (implemented play_sequence).
    """

    __test__ = False

    def __init__(
        self,
        env_factory,
        agent_network: ModularAgentNetwork,
        action_selector: BaseActionSelector,
        replay_buffer: ModularReplayBuffer,
        num_players: int,  # For compatibility with standard actor launch args
        config: Config,
        device: torch.device,
        name: str,
        test_types: Optional[List[BaseTestType]] = None,
        worker_id: int = 0,
    ):
        self.env_factory = env_factory
        self.agent_network = agent_network
        self.action_selector = action_selector
        self.replay_buffer = replay_buffer
        self.num_players = num_players
        self.config = config
        self.device = device
        self.name = name
        self.worker_id = worker_id

        # If test_types is NOT provided, it will be populated via Factory later
        # or during the first play_sequence call if needed.
        self.test_types = test_types or []

        # Initialize environment
        self.env = env_factory()
        self.actor_state = None  # For RNN/MCTS states

        # Initialize PolicySource
        use_search = hasattr(config, "search_backend") and getattr(
            config, "search_enabled", False
        )

        if use_search:
            from search.factory import SearchBackendFactory

            search_engine = SearchBackendFactory.create(config)
            self.policy_source = SearchPolicySource(
                search_engine, self.agent_network, config
            )
        else:
            self.policy_source = NetworkPolicySource(self.agent_network)

    def setup(self):
        """Initializes/resets the environment."""
        self.env = self.env_factory()
        self.actor_state = None
        self.agent_network.eval()

        if self.config.compilation.enabled:
            self.agent_network.compile(
                mode=self.config.compilation.mode,
                fullgraph=self.config.compilation.fullgraph,
            )

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Standard Actor API for updating parameters (epsilon, etc.)
        If params is a state_dict, we load it into agent_network.
        """
        if isinstance(params, dict):
            # Check if this looks like a state_dict or just hyperparameters
            if any(isinstance(v, (torch.Tensor, dict)) for v in params.values()):
                # CLEAN THE KEYS: Strip out any '_orig_mod.' prefixes
                # that might have come from the Learner's compiled network
                clean_params = {
                    k.replace("_orig_mod.", ""): v for k, v in params.items()
                }

                # strict=False is safer here as compilation might inject internal state keys
                self.agent_network.load_state_dict(clean_params, strict=False)
            else:
                self.action_selector.update_parameters(params)

    def select_test_action(self, state, info, env) -> Any:
        """Selects greedy action from the agent network."""
        with torch.inference_mode():
            # Use action_selector directly
            # Perform inference
            obs_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            # Perform inference via PolicySource
            result = self.policy_source.get_inference(
                obs=obs_tensor,
                info=info,
                agent_network=self.agent_network,
                exploration=False,
            )

            # Standardize masking
            if info is None:
                info = {}
            if "legal_moves" in info and "legal_moves_mask" not in info:
                action_tensor = result.action_dim
                assert (
                    action_tensor is not None
                ), "InferenceResult has no action tensor for mask shape"
                mask = torch.zeros(
                    action_tensor.shape, dtype=torch.bool, device=self.device
                )
                legal = info["legal_moves"]
                if isinstance(legal, (list, np.ndarray, torch.Tensor)):
                    mask[0, legal] = True
                info["legal_moves_mask"] = mask

            output = self.action_selector.select_action(
                result=result,
                info=info,
                exploration=False,
                actor_state=self.actor_state,
            )

            if isinstance(output, tuple) and len(output) == 2:
                action, self.actor_state = output
            else:
                action = output
            return action

    def play_sequence(self) -> Dict[str, Any]:
        """
        Standard Actor API called by Executors.
        Runs all test types and returns aggregated results.
        """
        # Ensure we have test types
        if not self.test_types:
            self.test_types = TestFactory.create_default_test_types(self.config)

        return self.run_tests()

    def run_tests(self) -> Dict[str, Any]:
        """Runs all configured test types and aggregates results."""
        all_results = {}
        for test_type in self.test_types:
            start_time = time.time()
            try:
                # Reset actor state before each test suite
                self.actor_state = None

                res = test_type.run(self, self.env)
                res["duration"] = time.time() - start_time
                all_results[test_type.name] = res
            except Exception as e:
                print(f"Error in test type '{test_type.name}': {e}")
                import traceback

                traceback.print_exc()

        return all_results


class TestFactory:
    """Factory for creating test strategies and Tester instances."""

    @staticmethod
    def create_default_test_types(
        config: Config, num_trials: Optional[int] = None
    ) -> List[BaseTestType]:
        """Creates standard test types based on game configuration."""
        test_types = []
        trials = num_trials if num_trials is not None else config.test_trials
        if config.game.multi_agent:
            test_types.append(SelfPlayTest("self_play", trials))
        else:
            test_types.append(StandardGymTest("standard", trials))
        return test_types

    @staticmethod
    def get_launch_args(
        config: Config,
        agent_network: ModularAgentNetwork,
        action_selector: BaseActionSelector,
        device: torch.device,
        name: str = "tester",
        test_types: Optional[List[BaseTestType]] = None,
    ) -> Tuple:
        """Returns the positional arguments expected by Tester.__init__."""
        return (
            config.game.env_factory,
            agent_network,
            action_selector,
            None,  # replay_buffer (Tester doesn't use it)
            config.game.num_players,
            config,
            device,
            name,
            test_types,
        )
