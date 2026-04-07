from typing import Dict, List, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from data.utils import discounted_cumulative_sums
from utils.utils import legal_moves_mask
from logging import warning

from data.processors.input_processors import InputProcessor
from data.processors.output_processors import OutputProcessor


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

        res = self._emit_oldest(player)
        self.n_step_buffers[player].popleft()
        return res

    def _emit_oldest(self, player):
        """
        Calculates the n-step return for the oldest transition in the buffer
        and returns the processed transition.
        """
        buffer = self.n_step_buffers[player]
        if not buffer:
            return None

        # Calculate N-Step Return
        # We look at the buffer to calculate discounted reward sum
        # The 'transition' to be returned is the oldest one in the deque (s_t)
        # The 'next_observation' will be the one from the newest transition (s_t+n)

        # 1. Calculate Discounted Reward
        final_reward = 0.0
        final_next_obs = buffer[-1].get("next_observations")
        final_next_info = buffer[-1].get("next_infos")
        final_next_legal_moves = buffer[-1].get("next_legal_moves")
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
                final_next_legal_moves = transition.get("next_legal_moves")
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
        head_transition["next_legal_moves"] = final_next_legal_moves
        head_transition[self.done_key] = final_done
        head_transition[self.terminated_key] = final_terminated
        head_transition[self.truncated_key] = final_truncated

        return head_transition

    def process_sequence(self, sequence, **kwargs):
        """
        Processes a sequence of transitions.
        """
        self.clear()
        transitions = kwargs.get("transitions")
        if transitions is None:
            transitions = self._sequence_to_transitions(sequence)
        processed_transitions = []

        for t in transitions:
            processed = self.process_single(**t)
            if processed is not None:
                processed_transitions.append(processed)

        # Flush remaining transitions if sequence is terminal
        is_done = sequence.done_history[-1] if sequence.done_history else False
        if is_done:
            active_players = set()
            for t in transitions:
                active_players.add(t.get("player", 0))

            for p in active_players:
                while self.n_step_buffers[p]:
                    res = self._emit_oldest(p)
                    if res:
                        processed_transitions.append(res)
                    self.n_step_buffers[p].popleft()

        return {"transitions": processed_transitions}

    def clear(self):
        self.n_step_buffers = [
            deque(maxlen=self.n_step) for _ in range(self.num_players)
        ]

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

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
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

            # To_play targets: valid at non-root states within the same game,
            # including terminal (terminal has a valid "whose turn" answer).
            # Zeroed at root (u=0) and post-terminal.
            is_consistent_tp = (u > 0) & (~post_done_mask[:, u]) & same_game[:, u]
            tp_indices = torch.clamp(raw_to_plays[:, u].long(), 0, self.num_players - 1)
            target_to_plays[range(batch_size), u, tp_indices] = 1.0
            target_to_plays[~is_consistent_tp, u] = 0

            target_dones[is_consistent, u] = raw_dones[is_consistent, u]
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
            has_valid_obs_mask=obs_valid_mask,
            has_valid_action_mask=dynamics_mask[:, : self.unroll_steps + 1],
            rewards=target_rewards,
            policies=target_policies,
            values=target_values,
            actions=target_actions,
            to_plays=target_to_plays,
            chance_codes=target_chances,
            dones=target_dones,
            is_same_game=same_game[:, : self.unroll_steps + 1],
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

        # Grounding: Ensure that any targets for padded steps (past game end) are explicitly 0.0
        # This is important for "safe absorbing states" logic where we want to learn V(s_absorbing) = 0
        target_values = target_values * valid_mask[:, :num_windows].float()

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