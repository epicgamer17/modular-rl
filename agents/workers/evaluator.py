import torch
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from modules.models.agent_network import AgentNetwork
from agents.action_selectors.policy_sources import BasePolicySource, NetworkPolicySource, SearchPolicySource
from agents.environments.adapters import BaseAdapter
from agents.action_selectors.selectors import BaseActionSelector
from agents.action_selectors.types import InferenceResult
from agents.workers.payloads import TaskRequest, TaskType, WorkerPayload
from agents.workers.actors import BaseActor


class BaseTestType(ABC):
    """
    Abstract base class for all test types.
    Defines the interface for running a specific type of evaluation.
    """

    def __init__(self, name: str, num_trials: int):
        self.name = name
        self.num_trials = num_trials

    @abstractmethod
    def run(self, actor: "EvaluatorActor") -> Dict[str, Any]:
        """
        Executes the test using the provided EvaluatorActor.

        Returns:
            Dictionary of metrics (score, etc.)
        """
        pass


class StandardGymTest(BaseTestType):
    """Evaluates a single-player environment."""

    def run(self, actor: "EvaluatorActor") -> Dict[str, Any]:
        scores = []
        lengths = []
        
        for _ in range(self.num_trials):
            obs, info = actor.adapter.reset()
            done = False
            trunc = False
            episode_reward = 0.0
            episode_length = 0
            
            while not (done or trunc):
                # 1. Inference
                result = actor.policy_source.get_inference(
                    obs, info, agent_network=actor.agent_network, exploration=False
                )
                
                # 2. Select action
                actions, _ = actor.action_selector.select_action(
                    result, info, exploration=False
                )
                
                # 3. Step environment
                obs, rewards, terminals, truncations, info = actor.adapter.step(actions)
                
                episode_reward += float(rewards.item())
                episode_length += 1
                done = bool(terminals.any().item())
                trunc = bool(truncations.any().item())
                
            scores.append(episode_reward)
            lengths.append(episode_length)
            
        return {
            "score": float(np.mean(scores)) if scores else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        }


class SelfPlayTest(BaseTestType):
    """
    Evaluates a multi-player environment where all players use the agent's network.
    """

    def run(self, actor: "EvaluatorActor") -> Dict[str, Any]:
        num_players = actor.num_players
        player_scores = {p: [] for p in range(num_players)}
        all_avg_scores = []
        total_lengths = []
        
        for _ in range(self.num_trials):
            obs, info = actor.adapter.reset()
            episode_length = 0
            done = False
            trunc = False
            
            # For self-play, we accumulate rewards for each player
            current_episode_rewards = np.zeros(num_players)
            
            while not (done or trunc):
                # 1. Inference
                result = actor.policy_source.get_inference(
                    obs, info, agent_network=actor.agent_network, exploration=False
                )
                
                # 2. Select action
                actions, _ = actor.action_selector.select_action(
                    result, info, exploration=False
                )
                
                # 3. Step environment
                obs, rewards, terminals, truncations, next_info = actor.adapter.step(actions)
                
                # Accumulate rewards for ALL players
                # info might contain 'all_rewards' (PettingZooAdapter pattern)
                all_rewards = next_info.get("all_rewards")
                if all_rewards is not None:
                    if isinstance(all_rewards, dict):
                        for p_idx, agent_id in enumerate(actor.adapter.agents):
                            current_episode_rewards[p_idx] += all_rewards.get(agent_id, 0.0)
                    elif torch.is_tensor(all_rewards):
                        current_episode_rewards += all_rewards.cpu().numpy()
                else:
                    # Fallback: only update the current player if all_rewards is missing
                    p_id = info.get("player_id", 0)
                    if torch.is_tensor(p_id): p_id = int(p_id.item())
                    current_episode_rewards[p_id] += float(rewards.item())
                
                info = next_info
                episode_length += 1
                done = bool(terminals.any().item())
                trunc = bool(truncations.any().item())
                
            for p in range(num_players):
                player_scores[p].append(current_episode_rewards[p])
            all_avg_scores.append(np.mean(current_episode_rewards))
            total_lengths.append(episode_length)
            
        results = {
            "score": float(np.mean(all_avg_scores)) if all_avg_scores else 0.0,
            "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0,
        }
        for p in range(num_players):
            results[f"p{p}_score"] = float(np.mean(player_scores[p]))
            
        return results


class VsAgentTest(BaseTestType):
    """
    Evaluates the agent against a specific fixed opponent.
    The agent takes `player_idx` and the opponent takes other positions.
    """

    def __init__(self, name: str, num_trials: int, opponent: Any, player_idx: int):
        super().__init__(name, num_trials)
        self.opponent = opponent
        self.player_idx = player_idx

    def run(self, actor: "EvaluatorActor") -> Dict[str, Any]:
        num_players = actor.num_players
        scores = []
        lengths = []
        
        for _ in range(self.num_trials):
            obs, info = actor.adapter.reset()
            episode_length = 0
            student_reward = 0.0
            done = False
            trunc = False
            
            while not (done or trunc):
                # 1. Identify current player
                curr_p = info.get("player_id", 0)
                if torch.is_tensor(curr_p): curr_p = int(curr_p.item())
                
                # 2. Get result from appropriate agent
                if curr_p == self.player_idx:
                    result = actor.policy_source.get_inference(
                        obs, info, agent_network=actor.agent_network, exploration=False
                    )
                else:
                    # Opponent inference
                    # Assuming opponent has an obs_inference method similar to AgentNetwork
                    output = self.opponent.obs_inference(obs, info=info, adapter=actor.adapter)
                    result = InferenceResult.from_inference_output(output)
                
                # 3. Select action
                actions, _ = actor.action_selector.select_action(
                    result, info, exploration=False
                )
                
                # 4. Step environment
                obs, rewards, terminals, truncations, next_info = actor.adapter.step(actions)
                
                # Accumulate student reward
                all_rewards = next_info.get("all_rewards")
                if all_rewards is not None:
                    if isinstance(all_rewards, dict):
                        agent_id = actor.adapter.agents[self.player_idx]
                        student_reward += all_rewards.get(agent_id, 0.0)
                    elif torch.is_tensor(all_rewards):
                        student_reward += all_rewards[self.player_idx].item()
                else:
                    if curr_p == self.player_idx:
                        student_reward += float(rewards.item())
                
                info = next_info
                episode_length += 1
                done = bool(terminals.any().item())
                trunc = bool(truncations.any().item())
                
            scores.append(student_reward)
            lengths.append(episode_length)
            
        return {
            "score": float(np.mean(scores)) if scores else 0.0,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        }


class EvaluatorActor(BaseActor):
    """
    Dedicated worker for evaluation and testing.
    Runs a suite of TestTypes and returns aggregated results.
    """

    def __init__(
        self,
        adapter_cls: Type[BaseAdapter],
        adapter_args: Tuple[Any, ...],
        network: AgentNetwork,
        policy_source: BasePolicySource,
        buffer: Optional[Any],  # Unused, for compatibility
        action_selector: BaseActionSelector,
        actor_device: str = "cpu",
        num_actions: Optional[int] = None,
        num_players: int = 1,
        test_types: Optional[List[BaseTestType]] = None,
        worker_id: int = 0,
        **kwargs,
    ):
        self.worker_id = worker_id
        device = torch.device(actor_device)
        self.adapter = adapter_cls(
            *adapter_args, device=device, num_actions=num_actions
        )

        self.agent_network = network
        self.policy_source = policy_source
        self.action_selector = action_selector
        self.num_players = num_players
        self.num_envs = self.adapter.num_envs
        
        # Default test types if none provided
        self.test_types = test_types or []
        if not self.test_types:
            # Simple default based on player count
            from agents.factories.evaluator import TestFactory
            # We delay factory call to avoid circular imports or just use a simple heuristic
            if num_players > 1:
                self.test_types = [SelfPlayTest("self_play", 5)]
            else:
                self.test_types = [StandardGymTest("standard", 5)]

    def setup(self) -> None:
        self.agent_network.eval()

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        if weights:
            clean_params = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
            self.agent_network.load_state_dict(clean_params, strict=False)

    def get_state(self) -> Dict[str, Any]:
        return {}

    def execute(self, request: TaskRequest) -> WorkerPayload:
        if request.task_type != TaskType.EVALUATE:
            raise ValueError(
                f"EvaluatorActor only supports {TaskType.EVALUATE}, got {request.task_type}"
            )

        # Allow overriding num_trials via request
        num_trials = request.batch_size
        results = self.run_tests(num_trials=num_trials, **request.kwargs)
        return WorkerPayload(worker_type=self.__class__.__name__, metrics=results)

    @torch.inference_mode()
    def run_tests(self, num_trials: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Runs all configured test types."""
        all_results = {}
        
        # If num_trials is passed, override it for all tests
        if num_trials is not None:
            for tt in self.test_types:
                tt.num_trials = num_trials

        for test_type in self.test_types:
            start_time = time.time()
            res = test_type.run(self)
            res["duration"] = time.time() - start_time
            all_results[test_type.name] = res
            
        # Calculate aggregated metrics BEFORE modifying all_results to avoid TypeError during iteration
        scores = [r["score"] for r in all_results.values() if "score" in r]
        lengths = [r["avg_length"] for r in all_results.values() if "avg_length" in r]

        # Standardize 'score' and 'avg_length' to be the average across all tests if not present
        if "score" not in all_results and scores:
            all_results["score"] = float(np.mean(scores))
        
        if "avg_length" not in all_results and lengths:
            all_results["avg_length"] = float(np.mean(lengths))

        return all_results
