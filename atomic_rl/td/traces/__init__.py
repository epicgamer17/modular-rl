"""
From Stream RL Paper:
"eligibility traces have been found to be effective primarily in tabular settings or with linear function approximation, while none of their deep-learning counterparts are known to perform well" - they do things to make it work in the paper, but something to think about in general with the below.
"""

from .accumulating import compute_accumulating_traces
from .replacing import compute_replacing_traces
from .true_online import compute_true_online_traces

__all__ = [
    "compute_accumulating_traces",
    "compute_replacing_traces",
    "compute_true_online_traces",
]
