from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Literal

@dataclass
class MarlHyperoptConfig:
    file_name: str
    eval_method: str  # "elo", "best_agent_elo", "test_agents_elo"
    best_agent: Any
    env_factory: Callable
    prep_params: Callable
    agent_class: Any
    agent_config: Callable
    game_config: Callable
    games_per_pair: int
    num_opps: int
    table: Any
    play_sequence: Callable
    checkpoint_interval: int = 100
    test_interval: int = 100
    test_trials: int = 10
    test_agents: List = field(default_factory=lambda: [])
    test_agent_weights: List[float] = field(default_factory=lambda: [])
    device: str = "cpu"

@dataclass
class SarlHyperoptConfig:
    file_name: str
    eval_method: Literal["final_score", "rolling_average", "final_score_rolling_average"]
    env_factory: Callable
    prep_params: Callable
    agent_class: Any
    agent_config: Callable
    game_config: Callable
    checkpoint_interval: int = 100
    test_interval: int = 100
    test_trials: int = 10
    last_n_rolling_avg: int = 10
    device: str = "cpu"

_MARL_CONFIG: Optional[MarlHyperoptConfig] = None
_SARL_CONFIG: Optional[SarlHyperoptConfig] = None

def set_sarl_config(config: SarlHyperoptConfig):
    global _SARL_CONFIG
    _SARL_CONFIG = config

def set_marl_config(config: MarlHyperoptConfig):
    global _MARL_CONFIG
    _MARL_CONFIG = config

def get_active_config():
    if _MARL_CONFIG is not None: return _MARL_CONFIG
    if _SARL_CONFIG is not None: return _SARL_CONFIG
    return None
