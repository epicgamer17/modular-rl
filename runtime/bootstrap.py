"""
Bootstrap script for the RL runtime.
Registers all built-in operators and specifications.
"""

from core.graph import (
    NODE_TYPE_SOURCE,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_EXPLORATION,
    NODE_TYPE_METRICS_SINK,
)
from runtime.operator_registry import register_operator
from runtime.signals import NoOp

# Import operator implementations and specs
from ops.control import SOURCE_SPEC, GET_FIELD_SPEC
from ops.buffer.query import op_replay_query, REPLAY_QUERY_SPEC, SAMPLE_BATCH_SPEC
from ops.rl.sync import op_target_sync, TARGET_SYNC_SPEC
from ops.rl.exploration import op_epsilon_greedy, EXPLORATION_SPEC
from ops.rl.metrics import op_metrics_sink, METRICS_SINK_SPEC
from ops.math.schedule import op_linear_decay, LINEAR_DECAY_SPEC
from ops.loss.math import op_mse_loss, MSE_LOSS_SPEC
from ops.math.reduce import (
    op_reduce_mean, 
    op_weighted_sum, 
    MEAN_SPEC, 
    WEIGHTED_SUM_SPEC, 
    REDUCE_MEAN_SPEC
)
from ops.math.clip import op_clip, CLIP_SPEC
from ops.rl.learner import (
    op_backward,
    op_grad_buffer,
    op_accumulate_grad,
    op_optimizer_step_every,
    op_optimizer_step,
    BACKWARD_SPEC,
    GRAD_BUFFER_SPEC,
    ACCUMULATE_GRAD_SPEC,
    OPTIMIZER_STEP_EVERY_SPEC,
    OPTIMIZER_SPEC
)
from ops.rl.q_learning import (
    op_q_values_single,
    op_q_forward,
    op_gather_action_q,
    op_bellman_target,
    Q_VALUES_SINGLE_SPEC,
    Q_FORWARD_SPEC,
    GATHER_ACTION_Q_SPEC,
    BELLMAN_TARGET_SPEC,
    TD_LOSS_SPEC
)
from ops.rl.policy import (
    op_policy_ratio,
    op_greedy_action,
    POLICY_RATIO_SPEC,
    GREEDY_ACTION_SPEC
)
from ops.rl.ppo_loss import (
    PPO_OPTIMIZER_SPEC,
    VALUE_LOSS_SPEC,
    SURROGATE_LOSS_SPEC
)
from ops.rl.distributions import (
    op_log_prob,
    op_entropy,
    LOG_PROB_SPEC,
    ENTROPY_SPEC
)
from runtime.io.transfer import register_transfer_operators

def bootstrap_runtime() -> None:
    """
    Registers all built-in operator specifications and execution functions.
    This should be called before executing any graphs.
    """
    # 2. Register core infrastructure operators
    register_operator(NODE_TYPE_SOURCE, lambda node, inputs, context=None: NoOp(), spec=SOURCE_SPEC)
    register_operator(NODE_TYPE_REPLAY_QUERY, op_replay_query, spec=REPLAY_QUERY_SPEC)
    register_operator("SampleBatch", op_replay_query, spec=SAMPLE_BATCH_SPEC)
    register_operator("GetField", lambda node, inputs, context=None: inputs.get("input").get(node.params["field"]) if isinstance(inputs.get("input"), dict) else getattr(inputs.get("input"), node.params["field"]), spec=GET_FIELD_SPEC)
    
    # 3. Register RL-specific operators
    register_operator(NODE_TYPE_TARGET_SYNC, op_target_sync, spec=TARGET_SYNC_SPEC)
    register_operator(NODE_TYPE_EXPLORATION, op_epsilon_greedy, spec=EXPLORATION_SPEC)
    register_operator(NODE_TYPE_METRICS_SINK, op_metrics_sink, spec=METRICS_SINK_SPEC)
    
    # 4. Register Math and Schedule operators
    register_operator("LinearDecay", op_linear_decay, spec=LINEAR_DECAY_SPEC)
    register_operator("MSELoss", op_mse_loss, spec=MSE_LOSS_SPEC)
    register_operator("Mean", op_reduce_mean, spec=MEAN_SPEC)
    register_operator("WeightedSum", op_weighted_sum, spec=WEIGHTED_SUM_SPEC)
    register_operator("ReduceMean", op_reduce_mean, spec=REDUCE_MEAN_SPEC)
    register_operator("Clip", op_clip, spec=CLIP_SPEC)
    
    # 5. Register Learner/Optimizer operators
    register_operator("Backward", op_backward, spec=BACKWARD_SPEC)
    register_operator("GradBuffer", op_grad_buffer, spec=GRAD_BUFFER_SPEC)
    register_operator("AccumulateGrad", op_accumulate_grad, spec=ACCUMULATE_GRAD_SPEC)
    register_operator("OptimizerStepEvery", op_optimizer_step_every, spec=OPTIMIZER_STEP_EVERY_SPEC)
    register_operator("Optimizer", op_optimizer_step, spec=OPTIMIZER_SPEC)
    register_operator("PPO_Optimizer", op_optimizer_step, spec=PPO_OPTIMIZER_SPEC)
    
    # 6. Register Q-Learning operators
    register_operator("QValuesSingle", op_q_values_single, spec=Q_VALUES_SINGLE_SPEC)
    register_operator("QForward", op_q_forward, spec=Q_FORWARD_SPEC)
    register_operator("QValuesBatch", op_q_forward, spec=Q_FORWARD_SPEC)
    register_operator("GatherActionQ", op_gather_action_q, spec=GATHER_ACTION_Q_SPEC)
    register_operator("BellmanTarget", op_bellman_target, spec=BELLMAN_TARGET_SPEC)
    register_operator("TDLoss", lambda node, inputs, context=None: NoOp(), spec=TD_LOSS_SPEC) # Stub if not implemented
    
    # 7. Register Policy and PPO operators
    register_operator("PolicyRatio", op_policy_ratio, spec=POLICY_RATIO_SPEC)
    register_operator("GreedyAction", op_greedy_action, spec=GREEDY_ACTION_SPEC)
    register_operator("ValueLoss", lambda node, inputs, context=None: NoOp(), spec=VALUE_LOSS_SPEC) # Stub
    register_operator("SurrogateLoss", lambda node, inputs, context=None: NoOp(), spec=SURROGATE_LOSS_SPEC) # Stub
    register_operator("LogProb", op_log_prob, spec=LOG_PROB_SPEC)
    register_operator("Entropy", op_entropy, spec=ENTROPY_SPEC)
    
    # 8. Register Data Transfer operators
    register_transfer_operators(register_operator)
