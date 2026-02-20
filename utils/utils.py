import random
import copy
import math
import matplotlib

matplotlib.use("Agg")

from typing import Iterable, Tuple
from datetime import datetime

import torch
from torch import Tensor

import numpy as np


def normalize_policies(policies: torch.float32):
    policy_sums = policies.sum(axis=-1, keepdims=True)
    policies = policies / policy_sums
    return policies


def legal_moves_mask(num_actions: int, legal_moves, device="cpu"):
    assert isinstance(legal_moves, list), "Legal moves should be a list"

    # FIX: Handle ragged lists (batches where games have different numbers of moves)
    # 1. Initialize empty boolean mask
    if len(legal_moves) > 0 and isinstance(legal_moves[0], list):
        # Batch processing
        batch_size = len(legal_moves)
        mask = torch.zeros((batch_size, num_actions), dtype=torch.bool).to(device)
        for i, moves in enumerate(legal_moves):
            if moves:  # Check if not empty
                mask[i, moves] = True
    else:
        # Single item processing
        mask = torch.zeros(num_actions, dtype=torch.bool).to(device)
        if legal_moves:
            mask[legal_moves] = True

    return mask.float()


def clip_low_prob_actions(actions: Tensor, low_prob: float = 0.01) -> Tensor:
    """
    Clip actions with probability lower than low_prob to 0 and re-normalize.
    """
    if low_prob == 0:
        return actions

    mask = actions < low_prob
    # FIX: Use torch.tensor(0.0) to ensure type compatibility
    actions = torch.where(mask, torch.tensor(0.0).to(actions.device), actions)

    # FIX: Re-normalize so probabilities sum to 1 again
    action_sums = actions.sum(axis=-1, keepdims=True) + 1e-7
    actions = actions / action_sums

    return actions


def get_legal_moves(info: dict | list[dict]):
    # FIX: Handle both single dict and list of dicts (Batch support)
    # print("info", info)
    if isinstance(info, list):
        legal_moves = [i.get("legal_moves", None) for i in info]
        for legal_list in legal_moves:
            assert len(legal_list) > 0
    else:
        legal_moves = [info.get("legal_moves", None)]
        # print("legal moves", legal_moves)
        if legal_moves[0] is None:
            return None
        assert len(legal_moves[0]) > 0
    return legal_moves


def normalize_images(image: Tensor) -> Tensor:
    """Preprocessing step to normalize image with 8-bit (0-255) color inplace.
    Modifys the original tensor

    Args:
        image (Tensor): An 8-bit color image

    Returns:
        Tensor: The tensor divided by 255
    """
    # Return a copy of the tensor divided by 255
    # FIX: Changed .div_(255) to / 255.0 to prevent corrupting the source memory
    return image / 255.0


def action_mask_to_legal_moves(action_mask):
    # print(action_mask)
    legal_moves = [i for i, x in enumerate(action_mask) if x == 1]
    # print(legal_moves)
    return legal_moves


def update_linear_schedule(
    final_value: float,
    total_steps: int,
    initial_value: float,
    current_step: int,
):
    # learning_rate = initial_value
    if initial_value < final_value:
        clamp_func = min
    else:
        clamp_func = max
    value = clamp_func(
        final_value,
        initial_value + ((final_value - initial_value) * (current_step / total_steps)),
    )
    return value


def update_inverse_sqrt_schedule(
    initial_value: float = None,
    current_step: int = None,
):
    return initial_value / math.sqrt(current_step + 1)


def epsilon_greedy_policy(
    q_values: list[float], info: dict, epsilon: float, wrapper=np.argmax
):
    if np.random.rand() < epsilon:
        # print("selecting a random move")
        if "legal_moves" in info:
            return random.choice(info["legal_moves"])
        else:
            q_values = q_values.reshape(-1)
            return random.choice(range(len(q_values)))
    else:
        # try:
        # print("using provided wrapper to select action")
        return wrapper(q_values, info)
    # except:
    #     return wrapper(q_values)


def add_dirichlet_noise(
    policy: list[float], dirichlet_alpha: float, exploration_fraction: float
):
    # MAKE ALPHAZERO USE THIS
    noise = np.random.dirichlet([dirichlet_alpha] * len(policy))
    frac = exploration_fraction
    for i, n in enumerate(noise):
        policy[i] = (1 - frac) * policy[i] + frac * n
    return policy


def augment_game(game, flip_y: bool = False, flip_x: bool = False, rot90: bool = False):
    # augmented_games[0] = rotate 90
    # augmented_games[1] = rotate 180
    # augmented_games[2] = rotate 270
    # augmented_games[3] = flip y (rotate 180 and flip x)
    # augmented_games[4] = rotate 90 and flip y (rotate 270 and flip x)
    # augmented_games[5] = rotate 180 and flip y (flip x)
    # augmented_games[6] = flip y and rotate 90 (rotate 270 and flip y) (rotate 90 and flip x)
    # augmented_games[7] = normal

    if (rot90 and flip_y) or (rot90 and flip_x):
        augemented_games = [copy.deepcopy(game) for _ in range(7)]
        for i in range(len(game.observation_history)):
            board = game.observation_history[i]
            policy = game.policy_history[i]
            augemented_games[0].observation_history[i] = np.rot90(board)
            augemented_games[0].policy_history[i] = np.rot90(policy)
            augemented_games[1].observation_history[i] = np.rot90(np.rot90(board))
            augemented_games[1].policy_history[i] = np.rot90(np.rot90(policy))
            augemented_games[2].observation_history[i] = np.rot90(
                np.rot90(np.rot90(board))
            )
            augemented_games[2].policy_history[i] = np.rot90(np.rot90(np.rot90(policy)))
            augemented_games[3].observation_history[i] = np.flipud(board)
            augemented_games[3].policy_history[i] = np.flipud(policy)
            augemented_games[4].observation_history[i] = np.flipud(np.rot90(board))
            augemented_games[4].policy_history[i] = np.flipud(np.rot90(policy))
            augemented_games[5].observation_history[i] = np.flipud(
                np.rot90(np.rot90(board))
            )
            augemented_games[5].policy_history[i] = np.flipud(
                np.rot90(np.rot90(policy))
            )
            augemented_games[6].observation_history[i] = np.rot90(np.flipud(board))
            augemented_games[6].policy_history[i] = np.rot90(np.flipud(policy))
    elif rot90 and not flip_y and not flip_x:
        augemented_games = [copy.deepcopy(game) for _ in range(3)]
        augemented_games[0].observation_history = [
            np.rot90(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.rot90(policy) for policy in game.policy_history
        ]
        augemented_games[1].observation_history = [
            np.rot90(np.rot90(board)) for board in game.observation_history
        ]
        augemented_games[1].policy_history = [
            np.rot90(np.rot90(policy)) for policy in game.policy_history
        ]
        augemented_games[2].observation_history = [
            np.rot90(np.rot90(np.rot90(board))) for board in game.observation_history
        ]
        augemented_games[2].policy_history = [
            np.rot90(np.rot90(np.rot90(policy)) for policy in game.policy_history)
        ]
    elif flip_y and not rot90 and not flip_x:
        augemented_games = [copy.deepcopy(game)]
        augemented_games[0].observation_history = [
            np.flipud(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.flipud(policy) for policy in game.policy_history
        ]

    elif flip_x and not rot90 and not flip_y:
        augemented_games = [copy.deepcopy(game) for _ in range(1)]
        augemented_games[0].observation_history = [
            np.fliplr(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.fliplr(policy) for policy in game.policy_history
        ]

    augemented_games.append(game)
    return augemented_games


def augment_board(
    board, policy, flip_y: bool = False, flip_x: bool = False, rot90: bool = False
):
    if (rot90 and flip_y) or (rot90 and flip_x):
        augemented_boards = [copy.deepcopy(board) for _ in range(7)]
        augmented_policies = [copy.deepcopy(policy) for _ in range(7)]
        augemented_boards[0] = np.rot90(board)
        augmented_policies[0] = np.rot90(policy)
        augemented_boards[1] = np.rot90(np.rot90(board))
        augmented_policies[1] = np.rot90(np.rot90(policy))
        augemented_boards[2] = np.rot90(np.rot90(np.rot90(board)))
        augmented_policies[2] = np.rot90(np.rot90(np.rot90(policy)))
        augemented_boards[3] = np.flipud(board)
        augmented_policies[3] = np.flipud(policy)
        augemented_boards[4] = np.flipud(np.rot90(board))
        augmented_policies[4] = np.flipud(np.rot90(policy))
        augemented_boards[5] = np.flipud(np.rot90(np.rot90(board)))
        augmented_policies[5] = np.flipud(np.rot90(np.rot90(policy)))
        augemented_boards[6] = np.rot90(np.flipud(board))
        augmented_policies[6] = np.rot90(np.flipud(policy))
    elif rot90 and not flip_y and not flip_x:
        augemented_boards = [copy.deepcopy(board) for _ in range(3)]
        augmented_policies = [copy.deepcopy(policy) for _ in range(3)]
        augemented_boards[0] = np.rot90(board)
        augmented_policies[0] = np.rot90(policy)
        augemented_boards[1] = np.rot90(np.rot90(board))
        augmented_policies[1] = np.rot90(np.rot90(policy))
        augemented_boards[2] = np.rot90(np.rot90(np.rot90(board)))
        augmented_policies[2] = np.rot90(np.rot90(np.rot90(policy)))
    elif flip_y and not rot90 and not flip_x:
        augemented_boards = [copy.deepcopy(board)]
        augmented_policies = [copy.deepcopy(policy)]
        augemented_boards[0] = np.flipud(board)
        augmented_policies[0] = np.flipud(policy)
    elif flip_x and not rot90 and not flip_y:
        augemented_boards = [copy.deepcopy(board)]
        augmented_policies = [copy.deepcopy(policy)]
        augemented_boards[0] = np.fliplr(board)
        augmented_policies[0] = np.fliplr(policy)
    augemented_boards.append(board)
    augmented_policies.append(policy)
    return augemented_boards, augmented_policies


def reward_clipping(reward: float, lower_bound: float = -1, upper_bound: float = 1):
    if reward < lower_bound:
        return lower_bound
    elif reward > upper_bound:
        return upper_bound
    return reward


def to_lists(l: list[Iterable]) -> list[Tuple]:
    """Convert a list of iterables to a zip of tuples

    Args:
        list (list[Iterable]): A list of iterables, e.g. [(1,1,1),(2,2,2),(3,3,3)]

    Returns:
        list[Tuple]: A list of tuples, i.e. [(1,2,3), (1,2,3), (1,2,3)]
    """

    return list(zip(*l))


def current_timestamp():
    return datetime.now().timestamp()


def isiterable(o):
    try:
        it = iter(o)
    except TypeError:
        return False
    return True


def tointlists(list):
    ret = []
    for x in list:
        if isiterable(x):
            ret.append(tointlists(x))
        else:
            ret.append(int(x))
    return ret


import time
from collections import deque


class StoppingCriteria:
    def __init__(self):
        pass

    def should_stop(self, details: dict) -> bool:
        return False


class TimeStoppingCriteria(StoppingCriteria):
    def __init__(self, max_runtime_sec=60 * 10):
        self.stop_time = time.time() + max_runtime_sec

    def should_stop(self, details: dict) -> bool:
        return time.time() > self.stop_time


class TrainingStepStoppingCritiera(StoppingCriteria):
    def __init__(self, max_training_steps=100000):
        self.max_training_steps = max_training_steps

    def should_stop(self, details: dict) -> bool:
        return details["training_step"] > self.max_training_steps


class EpisodesStoppingCriteria(StoppingCriteria):
    def __init__(self, max_episodes=100000):
        self.max_episodes = max_episodes

    def should_stop(self, details: dict) -> bool:
        return details["max_episodes"] > self.max_episodes


class AverageScoreStoppingCritera(StoppingCriteria):
    def __init__(self, min_avg_score: float, last_scores_length: int):
        self.min_avg_score = min_avg_score
        self.last_scores_length = last_scores_length
        self.last_scores = deque(maxlen=last_scores_length)

    def add_score(self, score: float):
        self.last_scores.append(score)

    def should_stop(self, details: dict) -> bool:
        if len(self.last_scores) < self.last_scores_length:
            return False

        return np.average(self.last_scores) < self.min_avg_score


class ApexLearnerStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.criterias: dict[str, StoppingCriteria] = {
            "time": TimeStoppingCriteria(max_runtime_sec=1.5 * 60 * 60),
            "training_step": TrainingStepStoppingCritiera(max_training_steps=10000),
            "avg_score": AverageScoreStoppingCritera(
                min_avg_score=15, last_scores_length=10
            ),
        }

    def should_stop(self, details: dict) -> bool:
        if self.criterias["time"].should_stop(details):
            return True

        if details["training_step"] < 10000:
            return False

        return self.criterias["training_step"].should_stop(details) or self.criterias[
            "avg_score"
        ].should_stop(details)

    def add_score(self, score: float):
        tc: AverageScoreStoppingCritera = self.criterias["avg_score"]
        tc.add_score(score)


def numpy_dtype_to_torch_dtype(np_dtype):
    """Converts a NumPy dtype to its corresponding PyTorch dtype."""
    # Create a temporary NumPy array with the desired dtype
    temp_np_array = np.empty([], dtype=np_dtype)
    # Convert it to a PyTorch tensor and extract its dtype
    return torch.from_numpy(temp_np_array).dtype


def recursive_batch(states: list | dict | torch.Tensor | tuple):
    if not states:
        return states

    # Check for mixed None and non-None states
    non_none_states = [s for s in states if s is not None]
    if not non_none_states:
        return states  # All elements are None

    if len(non_none_states) < len(states):
        # We have a mix. If we have dictionaries, we can recursively batch their keys.
        if isinstance(non_none_states[0], dict):
            # Use the union of keys from all non-None dictionaries
            all_keys = set()
            for s in non_none_states:
                all_keys.update(s.keys())
            return {
                k: recursive_batch(
                    [s.get(k) if s is not None else None for s in states]
                )
                for k in all_keys
            }
        # For other types (Tensor, Tuple), mixing with None is usually a bug.
        # We'll let it fall through to existing logic which will likely fail with a better error message,
        # or we could add an explicit assertion here.
        if isinstance(non_none_states[0], torch.Tensor):
            raise AssertionError(
                f"Cannot batch mixed None and Tensor values. Batch size: {len(states)}, Non-None count: {len(non_none_states)}"
            )

    if isinstance(states[0], dict):
        return {k: recursive_batch([s[k] for s in states]) for k in states[0].keys()}
    if isinstance(states[0], tuple):
        return tuple(
            recursive_batch([s[i] for s in states]) for i in range(len(states[0]))
        )
    if isinstance(states[0], torch.Tensor):
        # Handle LSTM states (L, 1, H) vs standard states
        if states[0].dim() == 3 and states[0].shape[1] == 1:
            return torch.cat(states, dim=1)
        # Handle states that already have a leading dimension of 1 (from SearchPolicy or previous unrolls)
        # Combine them into a single batch instead of stacking to (B, 1, ...)
        # TODO: is this safe?
        if states[0].dim() > 1 and states[0].shape[0] == 1:
            return torch.cat(states, dim=0)
        return torch.stack(states, dim=0)
    return states


def recursive_unbatch(batched_state: dict | torch.Tensor | tuple):
    if isinstance(batched_state, dict):
        keys = list(batched_state.keys())
        if not keys:
            return [{}]
        unbatched_values = [recursive_unbatch(batched_state[k]) for k in keys]
        # Find the intended batch size (max of unbatched sub-components)
        batch_size = max(len(v) for v in unbatched_values)
        return [
            {
                k: (
                    unbatched_values[i][j]
                    if len(unbatched_values[i]) > 1
                    else unbatched_values[i][0]
                )
                for i, k in enumerate(keys)
            }
            for j in range(batch_size)
        ]
    if isinstance(batched_state, tuple):
        unbatched_items = [recursive_unbatch(t) for t in batched_state]
        if not unbatched_items:
            return [()]
        batch_size = max(len(v) for v in unbatched_items)
        return [
            tuple(
                (
                    unbatched_items[i][j]
                    if len(unbatched_items[i]) > 1
                    else unbatched_items[i][0]
                )
                for i in range(len(batched_state))
            )
            for j in range(batch_size)
        ]
    if isinstance(batched_state, torch.Tensor):
        # Handle LSTM states (L, B, H)
        if batched_state.dim() == 3:
            return [batched_state[:, i : i + 1] for i in range(batched_state.shape[1])]
        # Return tensors with size-1 batch dimension to maintain consistency with initial_inference
        return [batched_state[i : i + 1] for i in range(batched_state.shape[0])]
    return [batched_state]
