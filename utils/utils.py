import random

import numpy as np
import torch
from torch import Tensor


def legal_moves_mask(num_actions: int, legal_moves, device="cpu"):
    assert isinstance(legal_moves, list), "Legal moves should be a list"

    if len(legal_moves) > 0 and isinstance(legal_moves[0], list):
        batch_size = len(legal_moves)
        mask = torch.zeros((batch_size, num_actions), dtype=torch.bool).to(device)
        for i, moves in enumerate(legal_moves):
            if moves:
                mask[i, moves] = True
    else:
        mask = torch.zeros(num_actions, dtype=torch.bool).to(device)
        if legal_moves:
            mask[legal_moves] = True

    return mask.float()


def get_legal_moves(info: dict | list[dict]):
    if isinstance(info, list):
        legal_moves = [i.get("legal_moves", None) for i in info]
        for legal_list in legal_moves:
            assert len(legal_list) > 0
    else:
        legal_moves = [info.get("legal_moves", None)]
        if legal_moves[0] is None:
            return None
        assert len(legal_moves[0]) > 0
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
