from typing import List
from compiler.rewrite import FusionRule

# Tier 1: Core Performance Fusions
GREEDY_ACTION_RULE = FusionRule(
    name="greedy_action",
    pattern=["QValuesSingle", "Argmax"],
    replacement="GreedyPolicy"
)

METRICS_FOLD_RULE = FusionRule(
    name="metrics_fold",
    pattern=["Mean", "MetricsSink"],
    replacement="MetricsFolded"
)

CLAMP_CAST_RULE = FusionRule(
    name="clamp_cast",
    pattern=["Clip", "Cast"],
    replacement="ClampedCast"
)

# Tier 2: Algorithm-Specific Optimization
PPO_ADVANTAGE_CHAIN_RULE = FusionRule(
    name="ppo_advantage_chain",
    pattern=["GAE", "Normalize", "PPOActorLoss"],
    replacement="PPOAdvantageLoss"
)

REPLAY_SAMPLE_PATH_RULE = FusionRule(
    name="replay_sample_path",
    pattern=["ReplayQuery", "Collate"],
    replacement="ReplaySample"
)

# Tier 3: Complex Architectural Fusions
FULL_POLICY_HEAD_RULE = FusionRule(
    name="full_policy_head",
    pattern=["Encoder", "PolicyHead", "Sample"],
    replacement="PolicyHeadFused"
)

RL_IR_FUSION_RULES: List[FusionRule] = [
    GREEDY_ACTION_RULE,
    METRICS_FOLD_RULE,
    CLAMP_CAST_RULE,
    PPO_ADVANTAGE_CHAIN_RULE,
    REPLAY_SAMPLE_PATH_RULE,
    FULL_POLICY_HEAD_RULE,
]
