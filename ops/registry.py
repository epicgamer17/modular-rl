from runtime.operator_registry import register_operator
from core.graph import NODE_TYPE_METRICS_SINK
from ops.control.loops import op_loop, op_minibatch_iterator
from ops.rl.q_learning import op_q_values_single, op_q_forward, op_gather_action_q, op_bellman_target
from ops.rl.policy import op_policy_actor, op_policy_forward, op_policy_ratio, op_greedy_action
from ops.rl.buffer import op_sample_all, op_sample_batch
from ops.loss.critic import op_td_loss, op_value_loss
from ops.loss.policy import op_surrogate_loss, op_entropy_loss
from ops.rl.distributions import op_log_prob, op_entropy
from ops.rl.advantage import op_advantage_estimation, op_gae, op_td_lambda, op_mc
from ops.rl.exploration import op_linear_decay, op_epsilon_greedy
from ops.control.access import op_get_field
from ops.math.reduce import op_reduce_mean, op_weighted_sum
from ops.rl.dagger import op_expert_actor
from ops.loss.supervised import op_sl_policy_loss, op_cross_entropy_loss
from ops.loss.math import op_mse_loss
from ops.rl.metrics import op_metrics_sink
from ops.rl.sync import op_target_sync
from ops.rl.learner import op_backward, op_grad_buffer, op_accumulate_grad, op_optimizer_step_every
from ops.math.clip import op_clip
from agents.ppo.operators import register_ppo_operators
from agents.dqn.operators import register_dqn_operators


_REGISTERED = False

def register_all_operators():
    """Registers all operators in the system."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    

    # Control
    register_operator("Loop", op_loop)
    register_operator("MinibatchIterator", op_minibatch_iterator)
    register_operator("GetField", op_get_field)
    
    # Q-Learning
    register_operator("QValuesSingle", op_q_values_single)
    register_operator("QForward", op_q_forward)
    register_operator("GatherActionQ", op_gather_action_q)
    register_operator("BellmanTarget", op_bellman_target)
    
    # Policy
    register_operator("PolicyForward", op_policy_actor)
    register_operator("PPO_Forward", op_policy_forward)
    register_operator("PolicyRatio", op_policy_ratio)
    register_operator("GreedyAction", op_greedy_action)
    register_operator("ExpertActor", op_expert_actor)
    
    # Buffer
    register_operator("SampleBatch", op_sample_all)
    register_operator("SampleBatchRandom", op_sample_batch)
    
    # Losses
    register_operator("TDLoss", op_td_loss)
    register_operator("ValueLoss", op_value_loss)
    register_operator("SurrogateLoss", op_surrogate_loss)
    register_operator("EntropyLoss", op_entropy_loss)
    register_operator("SLLoss", op_sl_policy_loss)
    register_operator("CrossEntropyLoss", op_cross_entropy_loss)
    register_operator("MSELoss", op_mse_loss)
    
    # RL Utilities
    register_operator("LogProb", op_log_prob)
    register_operator("Entropy", op_entropy)
    register_operator("AdvantageEstimation", op_advantage_estimation)
    register_operator("PPO_GAE", op_advantage_estimation)
    register_operator("GAE", op_gae)
    register_operator("TDLambda", op_td_lambda)
    register_operator("MC", op_mc)
    register_operator("LinearDecay", op_linear_decay)
    register_operator("Exploration", op_epsilon_greedy)
    register_operator("TargetSync", op_target_sync)
    register_operator(NODE_TYPE_METRICS_SINK, op_metrics_sink)
    
    # Math
    register_operator("ReduceMean", op_reduce_mean)
    register_operator("WeightedSum", op_weighted_sum)
    register_operator("Clip", op_clip)
    
    # Learner / Training
    register_operator("Backward", op_backward)
    register_operator("GradBuffer", op_grad_buffer)
    register_operator("AccumulateGrad", op_accumulate_grad)
    register_operator("OptimizerStepEvery", op_optimizer_step_every)
    
    # Agent Specific
    # These call their own logic but don't recurse back to register_all_operators
    register_ppo_operators()
    register_dqn_operators()

def register_ppo_operators_with_base():
    """Register PPO specific and all base operators."""
    register_all_operators()
    register_ppo_operators()

def register_dqn_operators_with_base():
    """Register DQN specific and all base operators."""
    register_all_operators()
    register_dqn_operators()

