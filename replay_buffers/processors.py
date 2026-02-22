from typing import List
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from modules.world_models.inference_output import LearningOutput
from utils.utils import legal_moves_mask
from replay_buffers.utils import discounted_cumulative_sums

# ==========================================
# Base Classes
# ==========================================


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
        pass

    def process_sequence(self, sequence, *args, **kwargs):
        """Optional hook for processing entire sequence objects."""
        raise NotImplementedError(
            "Sequence processing not implemented for this processor."
        )

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


class OutputProcessor(ABC):
    """
    Processes indices indices retrieved from the Sampler into a final batch.
    """

    @abstractmethod
    def process_batch(self, indices: list[int], buffers: dict, **kwargs):
        """
        Args:
            indices: List of indices selected by the Sampler.
            buffers: A dictionary reference to the ReplayBuffer's internal storage
                     (e.g., {'obs': self.observation_buffer, 'rew': self.reward_buffer}).
        Returns:
            batch: A dictionary containing the final tensors for training.
        """
        pass

    def clear(self):
        pass


# ==========================================
# Stacked Processors (Pipeline)
# ==========================================


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
            )
        for p in self.processors:
            data = p.process_single(**data)
            if data is None:
                return None

        return data

    def process_sequence(self, sequence, **kwargs):
        data = {"sequence": sequence, **kwargs}
        for p in self.processors:
            data = p.process_sequence(**data)
            if data is None:
                return None
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


class StackedOutputProcessor(OutputProcessor):
    """
    Chains multiple OutputProcessors.
    Each processor updates the 'batch' dictionary.
    """

    def __init__(self, processors: List[OutputProcessor]):
        self.processors = processors

    def process_batch(self, indices, buffers, batch=None, **kwargs):
        if batch is None:
            batch = {}

        for p in self.processors:
            # Processors should return a dict of new/updated keys
            # They receive the 'batch' so far to allow transformation (e.g. normalization)
            result = p.process_batch(indices, buffers, batch=batch, **kwargs)
            if result:
                batch.update(result)

        return batch

    def clear(self):
        for p in self.processors:
            p.clear()


# ==========================================
# Input Processors
# ==========================================


class IdentityInputProcessor(InputProcessor):
    """Pass-through processor."""

    def process_single(self, **kwargs):
        return kwargs


class LegalMovesInputProcessor(InputProcessor):
    """
    Extracts 'legal_moves' from 'info' or 'next_info' and creates a boolean mask.
    """

    def __init__(
        self,
        num_actions: int,
        info_key: str = "infos",
        output_key: str = "legal_moves_masks",
    ):
        self.num_actions = num_actions
        self.info_key = info_key
        self.output_key = output_key

    def process_single(self, **kwargs):
        info = kwargs.get(self.info_key, {})
        # Handle case where info might be None
        if info is None:
            info = {}

        moves = info.get("legal_moves", [])
        mask = legal_moves_mask(self.num_actions, moves)

        kwargs[self.output_key] = mask

        return kwargs


class ToPlayInputProcessor(InputProcessor):
    """
    Extracts 'player' or 'to_play' from 'info' or kwargs.
    """

    def __init__(
        self, num_players: int, info_key: str = "infos", output_key: str = "to_plays"
    ):
        self.num_players = num_players
        self.info_key = info_key
        self.output_key = output_key

    def process_single(self, **kwargs):
        # Check kwargs first, then info dict
        if "player" in kwargs:
            val = kwargs["player"]
        else:
            info = kwargs.get(self.info_key, {}) or {}
            val = info.get("player", 0)

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


class NStepInputProcessor(InputProcessor):
    """
    Handles N-Step return calculation.
    Accumulates transitions in a buffer and emits them when N steps are available.
    """

    def __init__(
        self,
        n_step: int,
        gamma: float,
        num_players: int = 1,
        reward_key="rewards",
        done_key="dones",
        terminated_key="terminated",
        truncated_key="truncated",
    ):
        self.n_step = n_step
        self.gamma = gamma
        self.num_players = num_players
        self.reward_key = reward_key
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key
        self.n_step_buffers = [deque(maxlen=n_step) for _ in range(num_players)]

    def process_single(self, **kwargs):
        # Determine player index
        player = kwargs.get("player", 0)

        # Store current step data
        self.n_step_buffers[player].append(kwargs)

        if len(self.n_step_buffers[player]) < self.n_step:
            return None

        # Calculate N-Step Return
        # We look at the buffer to calculate discounted reward sum
        # The 'transition' to be returned is the oldest one in the deque (s_t)
        # The 'next_observation' will be the one from the newest transition (s_t+n)

        buffer = self.n_step_buffers[player]

        # 1. Calculate Discounted Reward
        final_reward = 0.0
        final_next_obs = buffer[-1].get("next_observations")
        final_next_info = buffer[-1].get("next_infos")
        final_done = buffer[-1].get(self.done_key, False)
        final_terminated = buffer[-1].get(self.terminated_key, final_done)
        final_truncated = buffer[-1].get(self.truncated_key, False)

        # Iterate reversed from newest to oldest
        for transition in reversed(list(buffer)):
            r = transition.get(self.reward_key, 0.0)
            d = transition.get(self.done_key, False)

            # If a step was terminal, it cuts the n-step chain
            if d:
                final_reward = r
                final_next_obs = transition.get("next_observations")
                final_next_info = transition.get("next_infos")
                final_done = True
                final_terminated = transition.get(self.terminated_key, True)
                final_truncated = transition.get(self.truncated_key, False)
            else:
                final_reward = r + self.gamma * final_reward

        # 2. Prepare the output
        # The output is the oldest transition, but with updated reward/next_obs/done
        head_transition = buffer[0].copy()
        head_transition[self.reward_key] = final_reward
        head_transition["next_observations"] = final_next_obs
        head_transition["next_infos"] = final_next_info
        head_transition[self.done_key] = final_done
        head_transition[self.terminated_key] = final_terminated
        head_transition[self.truncated_key] = final_truncated

        return head_transition

    def clear(self):
        self.n_step_buffers = [
            deque(maxlen=self.n_step) for _ in range(self.num_players)
        ]


class SequenceTensorProcessor(InputProcessor):
    """
    Converts a complete Sequence object into tensor format for storage.
    Handles observation stacking, action/reward tensors, and legal move masks.
    """

    def __init__(self, num_actions: int, num_players: int, device="cpu"):
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device

    def process_single(self, **kwargs):
        raise NotImplementedError(
            "SequenceTensorProcessor only supports process_sequence."
        )

    def process_sequence(self, sequence, **kwargs):
        # 1. Prepare Observations
        obs_history = sequence.observation_history
        n_states = len(obs_history)
        n_transitions = len(sequence.action_history)
        if n_transitions + 1 != n_states:
            raise ValueError(
                "observation_history must have exactly one more entry than action_history"
            )
        if len(sequence.terminated_history) != n_states:
            raise ValueError(
                "Sequence terminated_history length must equal observation_history length"
            )
        if len(sequence.truncated_history) != n_states:
            raise ValueError(
                "Sequence truncated_history length must equal observation_history length"
            )
        if len(sequence.done_history) != n_states:
            raise ValueError(
                "Sequence done_history length must equal observation_history length"
            )
        obs_tensor = torch.from_numpy(np.stack(obs_history))  # .to(self.device)

        # 2. Prepare transition-aligned tensors (no algorithm-specific padding here).
        acts_t = torch.tensor(sequence.action_history, dtype=torch.float16)
        rews_t = torch.tensor(sequence.rewards, dtype=torch.float32)

        if sequence.policy_history:
            pols_t = (
                torch.stack(
                    [
                        p.detach() if torch.is_tensor(p) else torch.as_tensor(p)
                        for p in sequence.policy_history
                    ]
                )
                .cpu()
                .float()
            )
        else:
            pols_t = torch.empty((0, self.num_actions), dtype=torch.float32)

        # Values
        vals_t = torch.tensor(sequence.value_history, dtype=torch.float32)

        # To Plays
        tps_t = torch.tensor(
            [
                (
                    sequence.info_history[i] if i < len(sequence.info_history) else {}
                ).get("player", 0)
                for i in range(n_states)
            ],
            dtype=torch.int16,
        )

        # Chances
        chance_t = torch.tensor(
            [
                (
                    sequence.info_history[i] if i < len(sequence.info_history) else {}
                ).get("chance", 0)
                for i in range(n_states)
            ],
            dtype=torch.int16,
        ).unsqueeze(1)

        # Legal Moves Mask
        legal_masks_t = torch.stack(
            [
                legal_moves_mask(
                    self.num_actions,
                    (
                        sequence.info_history[i]
                        if i < len(sequence.info_history)
                        else {}
                    ).get("legal_moves", []),
                )
                for i in range(n_states)
            ]
        )

        # Episode end signals per state
        terminated_t = torch.tensor(sequence.terminated_history, dtype=torch.bool)
        truncated_t = torch.tensor(sequence.truncated_history, dtype=torch.bool)
        dones_t = torch.tensor(sequence.done_history, dtype=torch.bool)

        return {
            "observations": obs_tensor,
            "actions": acts_t,
            "rewards": rews_t,
            "policies": pols_t,
            "values": vals_t,
            "to_plays": tps_t,
            "chances": chance_t,
            "terminated": terminated_t,
            "truncated": truncated_t,
            "dones": dones_t,
            "legal_masks": legal_masks_t,
            "n_states": n_states,
        }


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

    def finish_trajectory(self, buffers, trajectory_slice, last_value=0):
        """
        Compute GAE advantages and returns for a trajectory segment.
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


# ==========================================
# Output Processors
# ==========================================


class StandardOutputProcessor(OutputProcessor):
    """Returns data indices directly."""

    def process_batch(self, indices, buffers, **kwargs):
        return {key: buf[indices] for key, buf in buffers.items()}


class NStepUnrollProcessor(OutputProcessor):
    """
    Handles window unrolling, validity masking, and N-step target calculation.
    Samples indices and creates unrolled sequences with N-step value bootstrapping.
    """

    def __init__(
        self,
        unroll_steps,
        n_step,
        gamma,
        num_actions,
        num_players,
        max_size,
        lstm_horizon_len=10,
        value_prefix=False,
        tau=0.3,
    ):
        self.unroll_steps = unroll_steps
        self.n_step = n_step
        self.gamma = gamma
        self.num_actions = num_actions
        self.num_players = num_players
        self.max_size = max_size
        self.lstm_horizon_len = lstm_horizon_len
        self.value_prefix = value_prefix
        self.tau = tau

    def process_batch(self, indices, buffers, **kwargs):
        # buffers dict should contain: obs, rew, val, pol, act, to_play, chance, game_id, legal_mask, training_step

        device = buffers["observations"].device
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        batch_size = len(indices)

        # 1. Define Window
        lookahead_window = self.unroll_steps + self.n_step
        offsets = torch.arange(
            0, lookahead_window + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        all_indices = (indices_tensor.unsqueeze(1) + offsets) % self.max_size

        # 2. Fetch Raw Data
        raw_rewards = buffers["rewards"][all_indices]
        raw_values = buffers["values"][all_indices]
        raw_policies = buffers["policies"][all_indices]
        raw_actions = buffers["actions"][all_indices]
        raw_to_plays = buffers["to_plays"][all_indices]
        raw_chances = buffers["chances"][all_indices]
        raw_game_ids = buffers["game_ids"][all_indices]
        raw_legal_masks = buffers["legal_masks"][all_indices]
        raw_terminated = (
            buffers["terminated"][all_indices]
            if "terminated" in buffers
            else buffers["dones"][all_indices]
        )
        raw_truncated = (
            buffers["truncated"][all_indices]
            if "truncated" in buffers
            else torch.zeros_like(raw_terminated)
        )
        raw_dones = raw_terminated | raw_truncated

        # 3. Validity Masks
        base_game_ids = raw_game_ids[:, 0].unsqueeze(1)
        same_game = raw_game_ids == base_game_ids

        # Calculate episode boundaries using dones (terminated/truncated)
        # We mask out any steps that occur AFTER a done signal in the sequence
        # cumsum gives us a mask of [0, 0, 1, 1, 1] if done happens at index 2
        cumulative_dones = torch.cumsum(raw_dones.float(), dim=1)

        # We want to mask steps *after* the done, not the done itself (which is a valid terminal state)
        # Shift cumsum right by 1: [0, 0, 0, 1, 1]
        post_done_mask = (
            torch.cat(
                [torch.zeros((batch_size, 1), device=device), cumulative_dones[:, :-1]],
                dim=1,
            )
            > 0
        )

        # Obs/Value Mask: Valid states (including terminal states), consistent with game ID and episode boundary
        obs_mask = same_game & (~post_done_mask)

        # Dynamics/Policy Mask: Valid transitions (excluding terminal states)
        # We cannot predict next state or policy FROM a terminal state
        dynamics_mask = obs_mask & (~raw_dones)

        # 5. Compute N-Step Targets
        target_values, target_rewards = self._compute_n_step_targets(
            batch_size,
            raw_rewards,
            raw_values,
            raw_to_plays,
            raw_terminated,
            raw_truncated,
            dynamics_mask,
            device,
        )

        # 6. Prepare Unroll Targets
        target_policies = torch.zeros(
            (batch_size, self.unroll_steps + 1, self.num_actions),
            dtype=torch.float32,
            device=device,
        )
        target_actions = torch.zeros(
            (batch_size, self.unroll_steps), dtype=torch.int64, device=device
        )
        target_to_plays = torch.zeros(
            (batch_size, self.unroll_steps + 1, self.num_players),
            dtype=torch.float32,
            device=device,
        )
        target_chances = torch.zeros(
            (batch_size, self.unroll_steps + 1, 1), dtype=torch.int64, device=device
        )
        target_dones = torch.ones(
            (batch_size, self.unroll_steps + 1), dtype=torch.bool, device=device
        )

        for u in range(self.unroll_steps + 1):
            is_consistent = dynamics_mask[:, u]

            target_policies[is_consistent, u] = raw_policies[is_consistent, u]
            target_policies[~is_consistent, u] = 1.0 / self.num_actions

            tp_indices = torch.clamp(raw_to_plays[:, u].long(), 0, self.num_players - 1)
            target_to_plays[range(batch_size), u, tp_indices] = 1.0
            target_to_plays[~is_consistent, u] = 0

            target_dones[is_consistent, u] = raw_dones[is_consistent, u]
            # If not consistent (different game or padding), treat as done
            target_dones[~is_consistent, u] = True

            target_chances[is_consistent, u, 0] = (
                raw_chances[is_consistent, u].squeeze(-1).long()
            )

            if u < self.unroll_steps:
                target_actions[:, u] = raw_actions[:, u].long()
                target_actions[~is_consistent, u] = int(
                    np.random.randint(0, self.num_actions)
                )

        # 7. Unroll Observations
        obs_indices = all_indices[:, : self.unroll_steps + 1]
        # Valid observations include terminal states
        obs_valid_mask = obs_mask[:, : self.unroll_steps + 1]

        unroll_observations = buffers["observations"][obs_indices].clone()

        for step in range(1, self.unroll_steps + 1):
            is_absorbing = ~obs_valid_mask[:, step]
            if is_absorbing.any():
                unroll_observations[is_absorbing, step] = unroll_observations[
                    is_absorbing, step - 1
                ]

        return dict(
            observations=buffers["observations"][indices_tensor],
            unroll_observations=unroll_observations,
            obs_mask=obs_valid_mask,
            action_mask=dynamics_mask[:, : self.unroll_steps + 1],
            rewards=target_rewards,
            policies=target_policies,
            values=target_values,
            actions=target_actions,
            to_plays=target_to_plays,
            chance_codes=target_chances,
            dones=target_dones,
            ids=buffers["ids"][indices_tensor].clone(),
            legal_moves_masks=buffers["legal_masks"][indices_tensor],
            indices=indices,
            training_steps=buffers["training_steps"][indices_tensor],
        )

    def _compute_n_step_targets(
        self,
        batch_size,
        raw_rewards,
        raw_values,
        raw_to_plays,
        raw_terminated,
        raw_truncated,
        valid_mask,
        device,
    ):
        """
        Vectorized N-step target calculation.
        """
        # 1. Setup Dimensions and Windows
        num_windows = self.unroll_steps + 1
        n_step = self.n_step
        gamma = self.gamma

        # Pre-compute gamma vector: [1, g, g^2, ..., g^(n-1)]
        # Shape: [1, 1, n_step]
        gammas = (
            gamma ** torch.arange(n_step, dtype=torch.float32, device=device)
        ).reshape(1, 1, n_step)

        # 2. Slice/Pad inputs to ensure we don't go out of bounds for the unfold
        # The 'raw' tensors typically have length >= num_windows + n_step
        # We need to create 'num_windows' windows of size 'n_step'.
        # The length required for this is: num_windows + n_step - 1.

        required_len = num_windows + n_step - 1
        raw_dones = raw_terminated | raw_truncated

        # Helper to pad if needed
        def safe_slice_unfold(tensor, length, size, step):
            if tensor.shape[1] < length:
                pad_len = length - tensor.shape[1]
                # Pad last dim (time)
                # pad format: (pad_left, pad_right, pad_top, pad_bottom...)
                # we want to pad dim 1. 2D tensor (B, L) -> (0, pad_len)
                # 3D tensor (B, L, C) -> (0, 0, 0, pad_len) ?? No, F.pad works on last dim first.
                if tensor.dim() == 2:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_len))
                elif tensor.dim() == 3:
                    # (B, L, C) -> pad L means pad dim 1.
                    # F.pad(dHW, dC, dL, dB) order?
                    # "Padding size: The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward."
                    # For (B, L, C): last is C (pad 0,0). Next is L (pad 0, pad_len).
                    tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))

            return tensor[:, :length].unfold(1, size, step)

        # [B, num_windows, n_step]
        rewards_windows = safe_slice_unfold(raw_rewards, required_len, n_step, 1)
        to_plays_windows = safe_slice_unfold(raw_to_plays, required_len, n_step, 1)
        dones_windows = safe_slice_unfold(raw_dones, required_len, n_step, 1)
        valid_windows = safe_slice_unfold(valid_mask, required_len, n_step, 1)

        # Current ToPlay for each window start (u)
        # We need raw_to_plays to be long enough for simple indexing too
        if raw_to_plays.shape[1] < num_windows:
            raw_to_plays = torch.nn.functional.pad(
                raw_to_plays, (0, num_windows - raw_to_plays.shape[1])
            )

        # Shape: [B, num_windows, 1]
        current_to_plays = raw_to_plays[:, :num_windows].unsqueeze(2)

        # 3. Compute Value Masks
        # Game Boundary Mask (from valid_mask) & Done Mask
        # We need to compute a mask that handles the "break" when a done occurs.
        # Original logic:
        #   valid = valid_mask & (~has_ended)
        #   has_ended |= done & valid

        # This implies:
        # - If done[k] is True, then mask[k] is VALID (we count the terminal reward).
        # - But mask[k+1] and onwards are INVALID.

        dones_float = dones_windows.float()
        # cumsum gives [0, 0, 1, 1] for dones [0, 0, 1, 0]
        # We want to forbid steps AFTER the first done.
        # Shift cumsum right by 1 to get "was done before this step?"
        was_done_before = torch.cat(
            [
                torch.zeros((batch_size, num_windows, 1), device=device),
                torch.cumsum(dones_float, dim=2)[:, :, :-1],
            ],
            dim=2,
        )

        # Check if "was done before" > 0 or if "was done before" depends on valid mask?
        # Typically simple cumsum is enough if we assume raw_dones are correct.

        # Valid Reward Steps:
        # 1. Original valid_mask is True (same game)
        # 2. No done happened BEFORE this step in the window
        valid_steps_mask = valid_windows & (was_done_before == 0)

        # 4. Compute Discounted Rewards
        # Player Sign: compared to current_to_play
        # [B, W, N]
        signs = torch.where(current_to_plays == to_plays_windows, 1.0, -1.0)

        # Weighted Rewards
        # sum(gamma^k * R_k * sign_k * valid_k)

        weighted_rewards = rewards_windows * gammas * signs * valid_steps_mask.float()

        # [B, num_windows]
        summed_rewards = weighted_rewards.sum(dim=2)

        # 5. Compute Bootstrap Value
        # Value at u + n_step (the boot_idx)
        # Boot indices: u + n_step
        boot_indices = torch.arange(n_step, n_step + num_windows, device=device)
        # Clamp to safeguard against OOB (though logic should prevent using it)
        safe_boot_indices = torch.clamp(boot_indices, max=raw_values.shape[1] - 1)

        # [B, num_windows]
        boot_values = raw_values[:, safe_boot_indices]
        boot_to_plays = raw_to_plays[:, safe_boot_indices]

        # Bootstrap Validity:
        # 1. Boot index must be within valid range (num_windows + n_step < max_size ideally)
        # 2. boot index valid_mask must be true
        # 3. Game must NOT have ended inside the n_step window.

        # Check if any done occurred in the valid part of the window
        # If valid_steps_mask has valid done, it's fine for reward, but kills bootstrap.
        # If any 'done' & 'valid' occurred in window -> no bootstrap.
        # We can sum up (dones_windows & valid_steps_mask).
        # If > 0, then we hit a done.
        terminated_windows = safe_slice_unfold(raw_terminated, required_len, n_step, 1)
        hit_terminated_in_window = (
            terminated_windows.float() * valid_steps_mask.float()
        ).sum(dim=2) > 0

        boot_is_valid = (
            valid_mask[:, safe_boot_indices]
            & (~hit_terminated_in_window)
            & (~raw_terminated[:, safe_boot_indices])
        )

        # Boot Sign
        boot_signs = torch.where(
            current_to_plays.squeeze(2) == boot_to_plays, 1.0, -1.0
        )

        boot_term = (gamma**n_step) * boot_values * boot_signs

        # Final Target Values
        target_values = summed_rewards + torch.where(
            boot_is_valid, boot_term, torch.tensor(0.0, device=device)
        )

        # 6. Target Rewards (Value Prefix or Instant)
        target_rewards = torch.zeros(
            (batch_size, num_windows), dtype=torch.float32, device=device
        )

        if self.value_prefix and self.lstm_horizon_len > 0:
            # Reimplement prefix accumulation loop (fast enough for small unroll_steps)
            prefix_sum = torch.zeros(batch_size, device=device)
            horizon_id = 0

            for u in range(1, num_windows):
                if horizon_id % self.lstm_horizon_len == 0:
                    prefix_sum = torch.zeros(batch_size, device=device)
                horizon_id += 1

                # Reward at u-1
                if (u - 1) < raw_rewards.shape[1]:
                    # We assume raw_rewards are valid for the prefix calculation
                    # (ignoring valid_mask logic here as per original implementation appearance
                    # or assuming consistency)
                    prefix_sum = prefix_sum + raw_rewards[:, u - 1]

                # Check if the TRANSITION to this node is valid (state u-1 must be non-terminal)
                is_valid = valid_mask[:, u - 1]
                target_rewards[is_valid, u] = prefix_sum[is_valid]
        else:
            # Standard: target_rewards[u] = raw_rewards[u-1]
            # The reward at step u is for the transition FROM state u-1,
            # so validity is determined by dynamics_mask at position u-1.
            t_rew = raw_rewards[:, : num_windows - 1]
            mask_slice = valid_mask[:, : num_windows - 1]  # positions 0..K-1

            # We assign to target_rewards[:, 1:]
            # But we must respect the mask
            # Safe way: fill zero, then assign masked

            # Slice match: [B, num_windows-1]
            target_slice = torch.zeros_like(target_rewards[:, 1:])
            target_slice[mask_slice] = t_rew[mask_slice]
            target_rewards[:, 1:] = target_slice

        return target_values, target_rewards


class AdvantageNormalizer(OutputProcessor):
    """
    Normalizes advantages and formats batches for policy gradient methods.
    """

    def process_batch(self, indices, buffers, **kwargs):
        # In PPO we usually sample the whole filled rollout and then minibatch in the learner.
        sl = slice(None) if indices is None else indices

        advantages = buffers["advantages"][sl].to(torch.float32)
        advantage_mean = advantages.mean()
        advantage_std = advantages.std()
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-10)

        return dict(
            observations=buffers["observations"][sl],
            actions=buffers["actions"][sl],
            advantages=normalized_advantages,
            returns=buffers["returns"][sl],
            log_probabilities=buffers["log_probabilities"][sl],
            legal_moves_masks=buffers["legal_moves_masks"][sl],
        )
