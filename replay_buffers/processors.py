from typing import List
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from modules.world_models.inference_output import LearningOutput
from utils.utils import legal_moves_mask

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
        """Optional hook for processing entire sequence objects (e.g. MuZero)."""
        raise NotImplementedError(
            "Batch/Sequence processing not implemented for this processor."
        )

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
        # 1. Standardize Input to Dict
        data = kwargs.copy()
        if args:
            # Handle legacy positional arguments from BaseBuffers
            # Assumes standard order: obs, info, act, rew, next_obs, next_info, done
            # keys = [
            #     "observation",
            #     "info",
            #     "action",
            #     "reward",
            #     "next_observation",
            #     "next_info",
            #     "done",
            # ]
            # for k, v in zip(keys, args):
            #     data[k] = v
            raise NotImplementedError(
                "Positional arguments are not supported in StackedInputProcessor."
            )
        # 2. Run Pipeline
        for p in self.processors:
            data = p.process_single(**data)
            if data is None:
                return None  # Pipeline halted (e.g. accumulating)

        return data

    def process_sequence(self, sequence, **kwargs):
        data = {"sequence": sequence, **kwargs}
        for p in self.processors:
            data = p.process_sequence(**data)
            if data is None:
                return None
        return data

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
    ):
        self.n_step = n_step
        self.gamma = gamma
        self.num_players = num_players
        self.reward_key = reward_key
        self.done_key = done_key
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
            else:
                final_reward = r + self.gamma * final_reward

        # 2. Prepare the output
        # The output is the oldest transition, but with updated reward/next_obs/done
        head_transition = buffer[0].copy()
        head_transition[self.reward_key] = final_reward
        head_transition["next_observations"] = final_next_obs
        head_transition["next_infos"] = final_next_info
        head_transition[self.done_key] = final_done

        return head_transition

    def clear(self):
        self.n_step_buffers = [
            deque(maxlen=self.n_step) for _ in range(self.num_players)
        ]


class MuZeroSequenceInputProcessor(InputProcessor):
    """
    Processes a complete Sequence object into tensors for MuZero storage.
    Extracted from: replay_buffers/muzero_replay_buffer.py
    """

    def __init__(self, num_actions: int, num_players: int, device="cpu"):
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device

    def process_single(self, **kwargs):
        """MuZeroSequenceInputProcessor only supports sequence processing."""
        raise NotImplementedError(
            "MuZeroSequenceInputProcessor only supports process_sequence."
        )

    def process_sequence(self, sequence, **kwargs):
        # 1. Prepare Observations
        obs_history = sequence.observation_history
        obs_tensor = torch.from_numpy(np.stack(obs_history))  # .to(self.device)

        # 2. Prepare & Pad Actions, Rewards, Policies
        # Actions
        acts_raw = torch.tensor(sequence.action_history, dtype=torch.float16)
        acts_pad = torch.zeros(1, dtype=torch.float16)
        acts_t = torch.cat([acts_raw, acts_pad])

        # Rewards
        rews_raw = torch.tensor(sequence.rewards, dtype=torch.float32)
        rews_pad = torch.zeros(1, dtype=torch.float32)
        rews_t = torch.cat([rews_raw, rews_pad])

        # Policies
        pols_raw = (
            torch.stack([p.detach() for p in sequence.policy_history]).cpu().float()
        )
        pols_pad = (
            torch.ones((1, self.num_actions), dtype=torch.float32) / self.num_actions
        )
        pols_t = torch.cat([pols_raw, pols_pad])

        # Values
        vals_t = torch.tensor(sequence.value_history, dtype=torch.float32)
        if len(vals_t) == len(sequence.action_history):
            vals_t = torch.cat([vals_t, torch.zeros(1, dtype=torch.float32)])

        # To Plays
        to_plays = [i.get("player", 0) for i in sequence.info_history]
        n_states = len(obs_history)
        if len(to_plays) < n_states:
            to_plays = to_plays + [0] * (n_states - len(to_plays))
        tps_t = torch.tensor(to_plays[:n_states], dtype=torch.int16)

        # Chances
        chances = [i.get("chance", 0) for i in sequence.info_history]
        if len(chances) < n_states:
            chances = chances + [0] * (n_states - len(chances))
        chance_t = torch.tensor(chances[:n_states], dtype=torch.int16).unsqueeze(1)

        # Legal Moves Mask
        legal_masks = []
        n_transitions = len(sequence.action_history)
        for i in range(n_transitions):
            moves = sequence.info_history[i].get("legal_moves", [])
            legal_masks.append(legal_moves_mask(self.num_actions, moves))
        # Terminal state mask: All False
        legal_masks.append(torch.zeros(self.num_actions, dtype=torch.bool))
        legal_masks_t = torch.stack(legal_masks)

        # Dones (Terminated/Truncated)
        # Assuming dones are stored in info or we can infer from sequence state
        dones = [i.get("done", False) for i in sequence.info_history]
        # Pad if needed
        if len(dones) < n_states:
            dones = dones + [False] * (n_states - len(dones))
        dones_t = torch.tensor(dones[:n_states], dtype=torch.bool)

        return {
            "observations": obs_tensor,
            "actions": acts_t,
            "rewards": rews_t,
            "policies": pols_t,
            "values": vals_t,
            "to_plays": tps_t,
            "chances": chance_t,
            "dones": dones_t,
            "legal_masks": legal_masks_t,
            "n_states": n_states,
        }


class PPOInputProcessor(InputProcessor):
    """
    Handles accumulation of trajectory for GAE calculation.
    """

    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # We need a temporary buffer or access to the main buffer to calculate GAE.
        # Assuming we can calculate GAE on the fly or post-trajectory like original code.

    def process_single(self, *args, **kwargs):
        # PPO usually stores directly, then post-processes at end of trajectory.
        return args

    def finish_trajectory(self, buffers, trajectory_slice, last_value=0):
        """
        Extracted from BasePPOReplayBuffer.finish_trajectory
        """
        rewards = torch.cat(
            (
                buffers["rewards"][trajectory_slice],
                torch.tensor([last_value], dtype=torch.float16),
            )
        )
        values = torch.cat(
            (
                buffers["values"][trajectory_slice],
                torch.tensor([last_value], dtype=torch.float16),
            )
        )

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        advantages = discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)
        returns = discounted_cumulative_sums(rewards, self.gamma)[:-1]

        return advantages, returns


# ==========================================
# Output Processors
# ==========================================


class StandardOutputProcessor(OutputProcessor):
    """Returns data indices directly."""

    def process_batch(self, indices, buffers, **kwargs):
        return {key: buf[indices] for key, buf in buffers.items()}


class MuZeroUnrollOutputProcessor(OutputProcessor):
    """
    Handles the complex window unrolling, validity masking, and N-step target calculation.
    Extracted from: replay_buffers/muzero_replay_buffer.py
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
        raw_dones = buffers["dones"][all_indices]

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
            raw_dones,
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
        raw_dones,
        valid_mask,
        device,
    ):
        target_values = torch.zeros(
            (batch_size, self.unroll_steps + 1), dtype=torch.float32, device=device
        )
        target_rewards = torch.zeros(
            (batch_size, self.unroll_steps + 1), dtype=torch.float32, device=device
        )
        lookahead_window = self.unroll_steps + self.n_step

        # Pre-allocate zero tensor to avoid redundant allocations in the loop
        zero = torch.tensor(0.0, device=device)

        value_prefix = 0.0
        horizon_id = 0  # Initialize horizon_id

        for u in range(self.unroll_steps + 1):
            is_valid_node = valid_mask[:, u]
            if u == 0:
                target_rewards[:, u] = 0.0
            else:
                if self.value_prefix:
                    # Reset accumulation periodically based on LSTM horizon
                    if horizon_id % self.lstm_horizon_len == 0:
                        value_prefix = 0.0
                    horizon_id += 1

                    # Get immediate reward (root u=0 has 0 reward)
                    current_reward = 0.0
                    reward_idx = u - 1
                    if reward_idx < raw_rewards.shape[1]:
                        current_reward = raw_rewards[:, reward_idx]

                    # Accumulate into prefix
                    value_prefix += current_reward
                    target_rewards[is_valid_node, u] = value_prefix[is_valid_node]
                else:
                    reward_idx = u - 1
                    target_rewards[is_valid_node, u] = (
                        raw_rewards[is_valid_node, reward_idx]
                        if reward_idx < raw_rewards.shape[1]
                        else 0.0
                    )

            computed_value = torch.zeros(batch_size, device=device)
            current_player = raw_to_plays[:, u]

            # Using done mask to determine game border/termination
            has_ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for k in range(self.n_step):
                r_idx = u + k
                if r_idx >= lookahead_window:
                    break

                # Valid step: Game ID match AND not explicitly ended via dones
                r_is_valid = valid_mask[:, r_idx] & (~has_ended)

                step_player = raw_to_plays[:, r_idx]
                sign = torch.where(current_player == step_player, 1.0, -1.0)
                reward_chunk = (self.gamma**k) * raw_rewards[:, r_idx] * sign
                computed_value += torch.where(r_is_valid, reward_chunk, zero)

                # If current step is done, mark as ended for *future* steps
                # (current reward is the final transition reward)
                has_ended = has_ended | (raw_dones[:, r_idx] & valid_mask[:, r_idx])

            boot_idx = u + self.n_step
            if boot_idx < lookahead_window:
                b_is_valid = valid_mask[:, boot_idx] & (~has_ended)
                boot_player = raw_to_plays[:, boot_idx]
                sign_boot = torch.where(current_player == boot_player, 1.0, -1.0)
                val_chunk = (
                    (self.gamma**self.n_step) * raw_values[:, boot_idx] * sign_boot
                )
                computed_value += torch.where(b_is_valid, val_chunk, zero)

            target_values[is_valid_node, u] = computed_value[is_valid_node]

        return target_values, target_rewards


class PPOOutputProcessor(OutputProcessor):
    """
    Handles normalization of advantages and formatting for PPO.
    Extracted from: BasePPOReplayBuffer.sample
    """

    def process_batch(self, indices, buffers, **kwargs):
        # In PPO 'indices' usually implies the whole buffer, or this is called after shuffling.

        # 1. Normalize Advantages
        adv_buffer = buffers["adv"]
        advantage_mean = torch.mean(torch.tensor(adv_buffer, dtype=torch.float32))
        advantage_std = torch.std(torch.tensor(adv_buffer, dtype=torch.float32))

        normalized_advantages = (adv_buffer - advantage_mean) / (advantage_std + 1e-10)

        # 2. Return Dict (Assuming PPO typically samples everything)
        # Note: If indices are provided, we slice.
        if indices is None:
            # Whole buffer
            sl = slice(None)
        else:
            sl = indices

        return dict(
            observations=buffers["obs"][sl],
            actions=buffers["act"][sl],
            advantages=normalized_advantages[sl],
            returns=buffers["ret"][sl],
            log_probabilities=buffers["log_prob"][sl],
            legal_moves_masks=buffers["legal_mask"][sl],
        )


class RSSMOutputProcessor(OutputProcessor):
    """
    Replicates RSSMReplayBuffer sampling: retrieves sequences of length L.
    """

    def __init__(self, batch_length, max_size):
        self.batch_length = batch_length
        self.max_size = max_size

    def process_batch(self, indices, buffers, **kwargs):
        # RSSM logic: sample sequence [i, i+L]
        batch_size = len(indices)

        # Create sequence offsets: [0, 1, ..., L-1]
        offsets = np.arange(self.batch_length)
        # Shape: (batch_size, batch_length)
        seq_indices = (np.array(indices)[:, None] + offsets[None, :]) % self.max_size

        results = {}
        for key, buf in buffers.items():
            # Retrieve data and possibly stack/reshape if needed
            data = buf[seq_indices]
            results[key] = data

        return results
