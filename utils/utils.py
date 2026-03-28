import random

import numpy as np
import torch
from torch import Tensor


def legal_moves_mask(num_actions: int, legal_moves, device="cpu"):
    if torch.is_tensor(legal_moves):
        # Case 1: Boolean mask [num_actions] or [B, num_actions]
        if legal_moves.dtype == torch.bool:
            if legal_moves.dim() == 1 and legal_moves.shape[0] == num_actions:
                return legal_moves.to(device=device, dtype=torch.float32)
            if legal_moves.dim() == 2 and legal_moves.shape[1] == num_actions:
                # Keep original batch dimension if present
                return legal_moves.to(device=device, dtype=torch.float32)

        # Handle tensors of indices (int/long) by converting to list for existing logic
        if legal_moves.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
            if legal_moves.dim() == 0:
                legal_moves = [legal_moves.item()]
            else:
                legal_moves = legal_moves.tolist()

    if isinstance(legal_moves, np.ndarray):
        legal_moves = legal_moves.tolist()

    assert isinstance(legal_moves, list), f"Legal moves should be a list, got {type(legal_moves)}"

    if len(legal_moves) > 0 and isinstance(legal_moves[0], list):
        batch_size = len(legal_moves)
        mask = torch.zeros((batch_size, num_actions), dtype=torch.bool, device=device)
        for i, moves in enumerate(legal_moves):
            if moves:
                mask[i, moves] = True
    else:
        mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
        if legal_moves:
            mask[legal_moves] = True

    return mask.float()


def get_legal_moves(info: dict | list[dict]):
    def _sanitize_moves(lm):
        if lm is None:
            return None
        if torch.is_tensor(lm):
            if lm.dim() == 0:
                return [lm.item()]
            if lm.dtype == torch.bool:
                # Convert boolean mask to list of indices
                if lm.dim() == 2 and lm.shape[0] == 1:
                    lm = lm.squeeze(0)
                # Ensure we are on CPU for .tolist() if it's large, 
                # but usually it's small if we are calling this in a Python loop.
                return torch.where(lm)[0].cpu().tolist()
            else:
                return lm.cpu().tolist()
        if isinstance(lm, np.ndarray):
            if lm.ndim == 0:
                return [lm.item()]
            return lm.tolist()
        if isinstance(lm, (int, np.integer)):
            return [int(lm)]
        return lm

    if isinstance(info, list):
        legal_moves = []
        for i in info:
            lm = i.get("legal_moves_mask", i.get("legal_moves", None))
            legal_moves.append(_sanitize_moves(lm))

        for legal_list in legal_moves:
            if legal_list is not None:
                assert len(legal_list) > 0, "Legal moves list is empty"
    else:
        lm = info.get("legal_moves_mask", info.get("legal_moves", None))
        lm = _sanitize_moves(lm)
        if lm is None:
            return None
        assert len(lm) > 0, "Legal moves list is empty"
        legal_moves = [lm]
    return legal_moves


def normalize_images(image: Tensor) -> Tensor:
    return image / 255.0


def action_mask_to_legal_moves(action_mask):
    legal_moves = [i for i, x in enumerate(action_mask) if x == 1]
    return legal_moves


def epsilon_greedy_policy(
    q_values: list[float], info: dict, epsilon: float, wrapper=np.argmax
):
    if np.random.rand() < epsilon:
        if "legal_moves" in info:
            return random.choice(info["legal_moves"])
        else:
            q_values = q_values.reshape(-1)
            return random.choice(range(len(q_values)))
    else:
        return wrapper(q_values, info)


def to_lists(l):
    return list(zip(*l))


def isiterable(o):
    try:
        iter(o)
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
    temp_np_array = np.empty([], dtype=np_dtype)
    return torch.from_numpy(temp_np_array).dtype
