"""
Transition class for DQN-style algorithms that store individual transitions
rather than complete game episodes.
"""

from dataclasses import dataclass
from typing import Any, Optional, List
import numpy as np


@dataclass
class Transition:
    """
    A single environment transition for DQN-style algorithms.

    This is the fundamental unit of experience for value-based RL algorithms
    like Rainbow DQN where transitions are stored and sampled independently.
    """

    observation: Any
    action: Any
    reward: float
    next_observation: Any
    done: bool
    info: Optional[dict] = None
    next_info: Optional[dict] = None
    metadata: Optional[dict] = None


@dataclass
class TransitionBatch:
    """
    A batch of transitions collected from one or more episodes.

    Used as the return type from DQNActor to allow collecting multiple
    transitions before returning to the trainer.
    """

    transitions: List[Transition]
    episode_stats: Optional[dict] = None  # e.g., episode_length, score

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self):
        return iter(self.transitions)
