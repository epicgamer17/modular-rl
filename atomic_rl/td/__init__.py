"""
Temporal Difference (TD) Learning Utilities.

This module is divided into two distinct paradigms:

1. Target Generators (`compute_*_target`):
   - Agnostic to function approximator (Works with Deep NNs).
   - Returns the target tensor to be used with standard PyTorch Loss functions and Optimizers.
   - Typically used in batched/replay settings (DQN, Actor-Critic).

2. Explicit Weight Updaters (`*_update`):
   - STRICTLY for Linear Function Approximation (V(s) = theta^T phi).
   - Manually applies the mathematical gradient step and returns the new weight vector.
   - Typically used in pure online, streaming RL settings without standard PyTorch Optimizers.

Reference: https://github.com/mohmdelsayed/streaming-drl
   The authors' streaming algorithms (stream_td.py, stream_ac_continuous.py, stream_dqn.py)
   compute TD targets and semi-gradient errors inline in the training loop; consult them
   for the released behavior this module mirrors.
"""

from . import targets
from . import traces
from . import updates

from .targets import (
    compute_v_td_target,
    compute_q_td_target,
    compute_categorical_q_td_target,
)
from .traces import (
    compute_accumulating_traces,
    compute_replacing_traces,
    compute_true_online_traces,
)
from .updates import (
    gtd0_update_,
    semi_gradient_td_update_,
    tdc_update_,
    true_online_td_update_,
)

__all__ = [
    "targets",
    "traces",
    "updates",
    "compute_v_td_target",
    "compute_q_td_target",
    "compute_categorical_q_td_target",
    "compute_accumulating_traces",
    "compute_replacing_traces",
    "compute_true_online_traces",
    "gtd0_update_",
    "semi_gradient_td_update_",
    "tdc_update_",
    "true_online_td_update_",
]

