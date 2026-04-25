"""DQN agent bootstrap.

The generic Optimizer op now lives in ops/rl/learner.py and is registered
globally in runtime/executor.py. This module only registers DQN-specific
metadata specs.
"""

from ops.loss.critic import op_td_loss


def register_dqn_operators():
    """Registers DQN metadata specs and operators."""
    from runtime.operator_registry import register_operator
    from agents.dqn.specs import register_dqn_specs

    register_operator("TDLoss", op_td_loss)
    register_dqn_specs()
