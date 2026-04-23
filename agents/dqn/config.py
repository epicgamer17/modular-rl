from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DQNConfig:
    obs_dim: int
    act_dim: int
    hidden_dim: int = 512
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_capacity: int = 50000
    batch_size: int = 128
    min_replay_size: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1000
    target_sync_frequency: int = 100
    model_handle: str = "online_q"
    target_handle: str = "target_q"
    buffer_id: str = "main"
