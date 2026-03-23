import torch
import numpy as np
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Type

from modules.models.agent_network import AgentNetwork
from agents.action_selectors.policy_sources import BasePolicySource, SearchPolicySource
from agents.environments.adapters import BaseAdapter
from agents.workers.actors import BaseActor, RolloutActor
from replay_buffers.modular_buffer import ModularReplayBuffer


class ReanalyzeActor(BaseActor):
    """
    Specifically for MuZero/EfficientZero re-analysis of stored sequences.
    Instead of stepping an environment, it fetches stored sequences from the buffer,
    re-runs MCTS with the current network, and updates the buffer with fresh targets.
    This effectively implements "MuZero Unplugged" style off-policy correction.
    """

    def __init__(
        self,
        network: AgentNetwork,
        search_policy_source: SearchPolicySource,
        buffer: ModularReplayBuffer,
        worker_id: int = 0,
    ):
        """
        Initializes the ReanalyzeActor.

        Args:
            network: The current AgentNetwork to use for MCTS.
            search_policy_source: Wrapped MCTS engine for re-generating policies/values.
            buffer: The ReplayBuffer to sample from and update.
            worker_id: Unique identifier for this worker.
        """
        self.worker_id = worker_id
        self.agent_network = network
        self.search_policy_source = search_policy_source
        self.buffer = buffer

    def setup(self) -> None:
        """Sets network to eval mode for re-analysis."""
        self.agent_network.eval()

    def update_parameters(self, state_dict: Dict[str, Any]) -> None:
        """Standard payload handling: updates weights for the re-analysis network."""
        if state_dict is None or not state_dict:
            return
            
        if any(isinstance(v, (torch.Tensor, dict)) for v in state_dict.values()):
            clean_params = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }
            self.agent_network.load_state_dict(clean_params, strict=False)


    def get_state(self) -> Dict[str, Any]:
        """Returns empty dict as this actor doesn't maintain rolling env metrics."""
        return {}

    @torch.inference_mode()
    def reanalyze(self, batch_size: int) -> Dict[str, Any]:
        """
        Main execution loop for re-analysis.
        Fetches 'batch_size' sequences, flattens them for vectorized MCTS, and writes targets back.
        """
        start_time = time.time()

        all_indices = []
        all_obs = []
        all_infos = []
        all_ids = []

        # 1. Fetch random game sequences from the buffer
        for _ in range(batch_size):
            seq_batch = self.buffer.sample_sequence()
            if seq_batch is None:
                continue

            idx = seq_batch["indices"]
            obs = seq_batch["observation"]  # [T, ...]
            ids = seq_batch.get("ids")  # MuZero consistency IDs

            all_indices.extend(idx)
            all_obs.append(obs)
            if ids is not None:
                all_ids.extend(ids.tolist())

            # Construct info dictionaries for each step in the sequence
            # Search requires 'player' and potentially 'legal_moves_mask'
            for i in range(len(idx)):
                step_info = {}
                if "player_id" in seq_batch:
                    step_info["player"] = int(seq_batch["player_id"][i])
                if "legal_moves_mask" in seq_batch:
                    step_info["legal_moves_mask"] = seq_batch["legal_moves_mask"][i]
                all_infos.append(step_info)

        if not all_obs:
            return {"reanalyzed_steps": 0}

        # 2. The Flattening Trick: [B, T, ...] -> [B*T, ...]
        # This allows running the fast MCTS backend on a single massive batch
        flat_obs = torch.cat(all_obs, dim=0).to(self.network.device)

        # 3. Generating Fresh Targets
        # The search_policy_source handles vectorized execution automatically
        result = self.search_policy_source.get_inference(
            flat_obs, all_infos, agent_network=self.network
        )

        # Extract target policies (visit counts) and root values
        new_policies = result.extras["target_policies"]
        new_values = result.value

        # 4. Push updated targets back into the buffer indices
        # IDs are used as a consistency guard to ensure we don't reanalyze evicted data
        self.buffer.reanalyze_sequence(
            indices=all_indices,
            new_policies=new_policies.cpu().numpy(),
            new_values=new_values.cpu().numpy(),
            ids=all_ids if all_ids else None,
        )

        duration = time.time() - start_time
        return {
            "reanalyzed_steps": len(all_indices),
            "duration": duration,
            "steps_per_second": len(all_indices) / duration,
        }


class DAggerActor(RolloutActor):
    """
    Data Aggregation (DAgger) specialized actor.
    Steers the environment with the Student's actions to encounter "hard" states,
    but labels the resulting transitions with the Expert's policy for supervised training.
    """

    def __init__(
        self,
        adapter_cls: Type[BaseAdapter],
        adapter_args: Tuple[Any, ...],
        student_network: AgentNetwork,
        expert_network: AgentNetwork,
        policy_source: BasePolicySource,  # Reference to student policy
        config: Any,
        buffer: ModularReplayBuffer,
        worker_id: int = 0,
    ):
        """
        Initializes the DAggerActor.

        Args:
            adapter_cls: Adapter class to use.
            adapter_args: Arguments for the adapter.
            student_network: The network being trained.
            expert_network: The fixed network providing labels.
            policy_source: Strategy for retrieving Student inferences.
            config: Algorithm configuration.
            buffer: Replay Buffer for storing labeled transitions.
            worker_id: Unique worker ID.
        """
        super().__init__(
            adapter_cls,
            adapter_args,
            student_network,
            policy_source,
            config,
            buffer,
            worker_id,
        )
        self.expert_network = expert_network

    @torch.inference_mode()
    def collect(self, num_steps: int) -> Dict[str, Any]:
        """
        Main execution loop.
        Steps with Student's action but records Expert's policy/logits as the target.
        """
        steps_this_call = 0
        start_time = time.time()

        while steps_this_call < num_steps:
            # 1. Student Inference (for control)
            result = self.policy_source.get_inference(
                self.obs, self.info, agent_network=self.agent_network
            )

            # 2. Expert Labeling Trick
            # Query the expert on the same observations the student is encountering
            expert_output = self.expert_network.obs_inference(self.obs)
            expert_probs = expert_output.policy.probs

            # 3. Use Student action to step environment
            if result.action is not None:
                actions = result.action
            else:
                actions = torch.multinomial(result.probs, 1).squeeze(-1)

            next_obs, rewards, terminals, truncations, infos = self.adapter.step(
                actions
            )

            # 4. Route Transitions (Label with Expert instead of Student)
            for i in range(self.num_envs):
                transition = {
                    "observation": next_obs[i].cpu().numpy(),
                    "action": actions[i].item(),
                    "reward": rewards[i].item(),
                    "terminated": terminals[i].item(),
                    "truncated": truncations[i].item(),
                    "policy": expert_probs[i].cpu().numpy(),  # THE LABEL
                    "value": (
                        result.value[i].item() if result.value is not None else None
                    ),
                }

                self.seq_manager.append(i, transition)

                if terminals[i] or truncations[i]:
                    seq = self.seq_manager.flush(i)
                    self.buffer.store_aggregate(seq)
                    self.episodes_completed += 1

                    self.seq_manager.append(
                        i,
                        {
                            "observation": next_obs[i].cpu().numpy(),
                            "terminated": False,
                            "truncated": False,
                        },
                    )

            self.obs = next_obs
            self.info = infos
            steps_this_call += self.num_envs
            self.total_steps += self.num_envs
            self.total_reward += rewards.sum().item()

        return {**self.get_state(), "duration": time.time() - start_time}


class NFSPActor(RolloutActor):
    """
    Neural Fictitious Self-Play (NFSP) specialized actor.
    Samples between RL (Best Response) and SL (Average Strategy).
    Tags transitions so they can be routed to correct SL or RL replay pools.
    """

    def __init__(
        self,
        adapter_cls: Type[BaseAdapter],
        adapter_args: Tuple[Any, ...],
        rl_network: AgentNetwork,
        avg_network: AgentNetwork,
        policy_source: BasePolicySource,  # This should handle anticipatory (eta) sampling
        config: Any,
        buffer: ModularReplayBuffer,
        worker_id: int = 0,
    ):
        """
        Initializes the NFSPActor.

        Args:
            adapter_cls: Adapter class.
            adapter_args: Arguments for the adapter.
            rl_network: Network used for Best Response (RL).
            avg_network: Network used for Average Strategy (SL).
            policy_source: PolicySource wrapping NFSP logic (η-sampling).
            config: Algorithm configuration.
            buffer: Replay Buffer handling tagged storage.
            worker_id: Unique worker ID.
        """
        super().__init__(
            adapter_cls,
            adapter_args,
            rl_network,
            policy_source,
            config,
            buffer,
            worker_id,
        )
        self.avg_network = avg_network
        self.expert_network = avg_network  # For generic evaluation if needed

    @torch.inference_mode()
    def collect(self, num_steps: int) -> Dict[str, Any]:
        """
        Steps environment and tags transitions with the active strategy.
        """
        steps_this_call = 0
        start_time = time.time()

        while steps_this_call < num_steps:
            # 1. Inference Pass (PolicySource handles eta-sampling between RL/SL)
            result = self.policy_source.get_inference(
                self.obs, self.info, agent_network=self.agent_network
            )

            # 2. Extract Strategy Tag and Action
            strategy = result.extras.get("policy_used", "best_response")

            if result.action is not None:
                actions = result.action
            else:
                actions = result.probs.argmax(dim=-1)

            # 3. Step Environment
            next_obs, rewards, terminals, truncations, infos = self.adapter.step(
                actions
            )

            # 4. Routing Trick: Tag every transition with the strategy that generated it
            for i in range(self.num_envs):
                transition = {
                    "observation": next_obs[i].cpu().numpy(),
                    "action": actions[i].item(),
                    "reward": rewards[i].item(),
                    "terminated": terminals[i].item(),
                    "truncated": truncations[i].item(),
                    "strategy": strategy,  # THE TAG: Buffer uses this for RL/SL routing
                }

                # SL targets require the RL policy logits/probs
                if strategy == "best_response":
                    transition["policy"] = result.probs[i].cpu().numpy()

                self.seq_manager.append(i, transition)

                if terminals[i] or truncations[i]:
                    seq = self.seq_manager.flush(i)
                    self.buffer.store_aggregate(seq)
                    self.episodes_completed += 1

                    self.seq_manager.append(
                        i,
                        {
                            "observation": next_obs[i].cpu().numpy(),
                            "terminated": False,
                            "truncated": False,
                        },
                    )

            self.obs = next_obs
            self.info = infos
            steps_this_call += self.num_envs
            self.total_steps += self.num_envs
            self.total_reward += rewards.sum().item()

        return {**self.get_state(), "duration": time.time() - start_time}
