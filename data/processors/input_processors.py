from typing import Dict, List, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from data.utils import discounted_cumulative_sums
from utils.utils import legal_moves_mask
from logging import warning



class InputProcessor(ABC):
    """
    Processes data BEFORE it is written to the Writer/Storage.
    """

    @abstractmethod
    def process_single(self, *args, **kwargs):
        """
        Processes a single transition.
        Returns:
            processed_data: Data ready to be stored (or None if accumulating).
        """
        pass  # pragma: no cover

    def process_sequence(self, sequence, **kwargs):
        """Optional hook for processing entire sequence objects."""
        # Default behavior: iterate over transitions and apply process_single
        transitions = kwargs.get("transitions")
        if transitions is None:
            transitions = self._sequence_to_transitions(sequence)

        processed = []
        for t in transitions:
            res = self.process_single(**t)
            if res is not None:
                processed.append(res)
        return {"transitions": processed}

    def _sequence_to_transitions(self, sequence):
        """
        Helper to convert a Sequence object into a list of transition dictionaries
        compatible with process_single.
        """
        transitions = []
        for i in range(len(sequence.action_history)):
            t = {
                "observations": sequence.observation_history[i],
                "actions": sequence.action_history[i],
                "rewards": (
                    float(sequence.rewards[i]) if i < len(sequence.rewards) else 0.0
                ),
                "next_observations": sequence.observation_history[i + 1],
                "terminated": bool(sequence.terminated_history[i + 1]),
                "truncated": bool(sequence.truncated_history[i + 1]),
                "dones": bool(sequence.done_history[i + 1]),
                "player": (
                    sequence.player_id_history[i]
                    if i < len(sequence.player_id_history)
                    else 0
                ),
                "values": (
                    sequence.value_history[i]
                    if i < len(sequence.value_history)
                    else 0.0
                ),
                "policies": (
                    sequence.policy_history[i]
                    if i < len(sequence.policy_history)
                    else None
                ),
                "legal_moves": (
                    sequence.legal_moves_history[i]
                    if i < len(sequence.legal_moves_history)
                    else None
                ),
                "next_legal_moves": (
                    sequence.legal_moves_history[i + 1]
                    if i + 1 < len(sequence.legal_moves_history)
                    else None
                ),
            }
            # Add next_infos if available (usually not in Sequence but for completeness)
            if hasattr(sequence, "next_infos") and i < len(sequence.next_infos):
                t["next_infos"] = sequence.next_infos[i]

            transitions.append(t)
        return transitions

    def finish_trajectory(self, buffers, trajectory_slice, **kwargs):
        """
        Optional hook called when a trajectory ends.
        Override to compute trajectory-level computations (e.g., GAE).
        Returns:
            Optional dict of computed values to store in buffer.
        """
        return None

    def clear(self):
        pass

class StackedInputProcessor(InputProcessor):
    """
    Chains multiple InputProcessors sequentially.
    Output of processor i is passed as input to processor i+1.
    """

    def __init__(self, processors: List[InputProcessor]):
        self.processors = processors

    def process_single(self, *args, **kwargs):
        data = kwargs.copy()
        if args:
            raise NotImplementedError(
                "Positional arguments are not supported in StackedInputProcessor."
            )  # pragma: no cover
        for p in self.processors:
            data = p.process_single(**data)
            if data is None:
                return None

        return data

    def process_sequence(self, sequence, **kwargs):
        data = {"sequence": sequence, **kwargs}
        for p in self.processors:
            result = p.process_sequence(**data)
            if result is None:
                return None
            data.update(result)
        return data

    def finish_trajectory(self, buffers, trajectory_slice, **kwargs):
        """
        Calls finish_trajectory on all processors, aggregating results.
        """
        result = {}
        for p in self.processors:
            traj_result = p.finish_trajectory(buffers, trajectory_slice, **kwargs)
            if traj_result:
                result.update(traj_result)
        return result

    def get_processor(self, processor_type):
        """
        Find a processor by type in the stack.
        Returns the first matching processor or None.
        """
        for p in self.processors:
            if isinstance(p, processor_type):
                return p
        return None

    def clear(self):
        for p in self.processors:
            p.clear()

class IdentityInputProcessor(InputProcessor):
    """Pass-through processor."""

    def process_single(self, **kwargs):
        return kwargs

class LegalMovesMaskProcessor(InputProcessor):
    """
    Creates a boolean mask from a list of legal moves.
    """

    def __init__(
        self,
        num_actions: int,
        input_key: str = "legal_moves",
        output_key: str = "legal_moves_masks",
    ):
        self.num_actions = num_actions
        self.input_key = input_key
        self.output_key = output_key

    def process_single(self, **kwargs):
        legal_moves = kwargs.get(self.input_key, [])
        if legal_moves is None:
            legal_moves = []
        mask = legal_moves_mask(self.num_actions, legal_moves)
        kwargs[self.output_key] = mask
        return kwargs

class ToPlayInputProcessor(InputProcessor):
    """
    Extracts 'player_id' from kwargs.
    """

    def __init__(
        self,
        num_players: int,
        input_key: str = "player",
        output_key: str = "to_plays",
    ):
        self.num_players = num_players
        self.input_key = input_key
        self.output_key = output_key

    def process_single(self, **kwargs):
        val = kwargs.get(self.input_key, 0)
        kwargs[self.output_key] = val
        return kwargs

class FilterKeysInputProcessor(InputProcessor):
    """
    Filters the input dictionary to only include specific keys (whitelist).
    """

    def __init__(self, whitelisted_keys: List[str]):
        self.whitelisted_keys = whitelisted_keys

    def process_single(self, **kwargs):
        return {k: v for k, v in kwargs.items() if k in self.whitelisted_keys}

class TerminationFlagsInputProcessor(InputProcessor):
    """
    Ensures 'terminated'/'truncated' flags are present for transition pipelines.
    Defaults are conservative and backward-compatible with older call sites.
    """

    def __init__(
        self,
        done_key: str = "dones",
        terminated_key: str = "terminated",
        truncated_key: str = "truncated",
    ):
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key

    def process_single(self, **kwargs):
        done = bool(kwargs.get(self.done_key, False))
        kwargs[self.terminated_key] = bool(kwargs.get(self.terminated_key, done))
        kwargs[self.truncated_key] = bool(kwargs.get(self.truncated_key, False))
        return kwargs

class GAEProcessor(InputProcessor):
    """
    Computes Generalized Advantage Estimation (GAE) at trajectory end.
    Passes through single steps unchanged, then finalizes with GAE calculation.
    """

    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def process_single(self, *args, **kwargs):
        return kwargs

    def process_sequence(self, sequence, **kwargs):
        """
        Computes GAE over a full sequence of transitions.
        """
        transitions = kwargs.get("transitions")
        if transitions is None:
            transitions = self._sequence_to_transitions(sequence)

        if transitions is None or len(transitions) == 0:
            return {"transitions": []}

        # Extract rewards and values
        rewards_np = np.array(
            [t.get("rewards", 0.0) for t in transitions], dtype=np.float32
        )
        values_np = np.array(
            [t.get("values", 0.0) for t in transitions], dtype=np.float32
        )

        # PPO usually expects a last value for the next state of the final transition.
        # Check if sequence has value_history that is n+1 long
        if (
            sequence is not None
            and hasattr(sequence, "value_history")
            and len(sequence.value_history) > len(transitions)
        ):
            last_value = sequence.value_history[-1]
        else:
            last_value = 0.0

        dones_np = np.array([t.get("dones", False) for t in transitions], dtype=bool)

        rewards_pad = np.append(rewards_np, last_value)
        values_pad = np.append(values_np, last_value)

        # Vectorized GAE
        deltas = (
            rewards_pad[:-1]
            + self.gamma * values_pad[1:] * (~dones_np)
            - values_pad[:-1]
        )

        advantages = discounted_cumulative_sums(
            deltas, self.gamma * self.gae_lambda
        ).copy()
        returns = discounted_cumulative_sums(rewards_pad, self.gamma).copy()[:-1]

        adv_list = advantages.tolist()
        ret_list = returns.tolist()
        for t, adv, ret in zip(transitions, adv_list, ret_list):
            t["advantages"] = adv
            t["returns"] = ret

        return {"transitions": transitions}

    def finish_trajectory(self, buffers, trajectory_slice, last_value=0):
        """
        Compute GAE advantages and returns for a trajectory segment in old single-step mode.
        """
        rewards = torch.cat(
            (
                buffers["rewards"][trajectory_slice],
                torch.tensor([last_value], dtype=torch.float32),
            )
        )
        values = torch.cat(
            (
                buffers["values"][trajectory_slice],
                torch.tensor([last_value], dtype=torch.float32),
            )
        )

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        deltas_np = deltas.detach().cpu().numpy()
        rewards_np = rewards.detach().cpu().numpy()

        advantages = torch.from_numpy(
            discounted_cumulative_sums(deltas_np, self.gamma * self.gae_lambda).copy()
        ).to(torch.float32)
        returns = torch.from_numpy(
            discounted_cumulative_sums(rewards_np, self.gamma).copy()
        ).to(torch.float32)[:-1]

        return {"advantages": advantages, "returns": returns}