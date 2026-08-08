from .value_target import compute_v_td_target
from .q_target import compute_q_td_target, compute_categorical_q_td_target

__all__ = [
    "compute_v_td_target",
    "compute_q_td_target",
    "compute_categorical_q_td_target",
]
