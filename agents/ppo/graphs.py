from core.graph import (
    Graph,
    NODE_TYPE_SOURCE,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_METRICS_SINK,
)
from core.schema import Schema, Field, TensorSpec, TAG_ON_POLICY, TAG_ORDERED
from .config import PPOConfig

def create_ppo_schema(config: PPOConfig) -> Schema:
    """Create the PPO data schema."""
    obs_spec = TensorSpec(shape=(config.obs_dim,), dtype="float32")
    act_spec = TensorSpec(shape=(), dtype="int64")

    return Schema(
        fields=[
            Field("obs", obs_spec),
            Field("action", act_spec),
            Field("reward", TensorSpec(shape=(), dtype="float32")),
            Field("next_obs", obs_spec),
            Field("terminated", TensorSpec(shape=(), dtype="bool")),
            Field("truncated", TensorSpec(shape=(), dtype="bool")),
            Field("log_prob", TensorSpec(shape=(), dtype="float32")),
            Field("policy_version", TensorSpec(shape=(), dtype="int64")),
        ]
    )

def create_interact_graph(config: PPOConfig) -> Graph:
    """Create the PPO interaction graph."""
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    graph.add_node(
        "actor", 
        "PolicyForward", 
        params={"model_handle": config.model_handle}, 
        tags=[TAG_ON_POLICY]
    )
    graph.add_edge("obs_in", "actor", dst_port="obs")
    return graph

def create_train_graph(config: PPOConfig) -> Graph:
    """Create the PPO training graph using primitive operators."""
    graph = Graph()
    
    # Source for the pre-processed minibatch
    graph.add_node("traj_in", NODE_TYPE_SOURCE)
    
    # 1. Forward Pass
    graph.add_node(
        "forward", 
        "PPO_Forward", 
        params={
            "model_handle": config.model_handle,
            "activation_checkpoint": config.activation_checkpoint
        }
    )
    
    # 2. Log Probabilities
    graph.add_node("log_prob", "LogProb")
    
    # 3. Ratio
    graph.add_node("ratio", "PolicyRatio")
    
    # 4. Clip
    graph.add_node("clip", "Clip", params={"eps": config.clip_coef})
    
    # 5. Surrogate Loss
    graph.add_node("surr_loss", "SurrogateLoss")
    
    # 6. Value Loss
    graph.add_node(
        "val_loss", 
        "ValueLoss", 
        params={
            "eps": config.clip_coef,
            "clip": config.value_clip
        }
    )
    
    # 7. Entropy
    graph.add_node("entropy", "Entropy")
    
    # 8. Total Loss (Weighted Sum)
    graph.add_node(
        "total_loss", 
        "WeightedSum", 
        params={
            "surr": 1.0,
            "val": config.vf_coef,
            "ent": -config.ent_coef
        }
    )
    
    # 9. Optimizer Step
    graph.add_node(
        "opt", 
        "PPO_Optimizer", 
        params={
            "optimizer_handle": config.optimizer_handle,
            "model_handle": config.model_handle
        }
    )
    
    # 10. Metrics Sink
    graph.add_node(
        "metrics",
        NODE_TYPE_METRICS_SINK,
        params={"log_frequency": 32, "buffer_id": config.buffer_id},
    )

    # Wiring
    # Forward Pass
    graph.add_edge("traj_in", "forward", src_port="obs", dst_port="obs")
    
    # Log Prob
    graph.add_edge("forward", "log_prob", src_port="logits", dst_port="logits")
    graph.add_edge("traj_in", "log_prob", src_port="action", dst_port="action")
    
    # Ratio
    graph.add_edge("log_prob", "ratio", dst_port="new_log_prob")
    graph.add_edge("traj_in", "ratio", src_port="log_prob", dst_port="old_log_prob")
    
    # Clip
    graph.add_edge("ratio", "clip", dst_port="x")
    
    # Surrogate Loss
    graph.add_edge("ratio", "surr_loss", dst_port="ratio")
    graph.add_edge("clip", "surr_loss", dst_port="clipped_ratio")
    graph.add_edge("traj_in", "surr_loss", src_port="advantages", dst_port="advantages")
    
    # Value Loss
    graph.add_edge("forward", "val_loss", src_port="values", dst_port="values")
    graph.add_edge("traj_in", "val_loss", src_port="returns", dst_port="returns")
    graph.add_edge("traj_in", "val_loss", src_port="value", dst_port="old_values")
    
    # Entropy
    graph.add_edge("forward", "entropy", src_port="logits", dst_port="logits")
    
    # Total Loss
    graph.add_edge("surr_loss", "total_loss", dst_port="surr")
    graph.add_edge("val_loss", "total_loss", dst_port="val")
    graph.add_edge("entropy", "total_loss", dst_port="ent")
    
    # Optimizer & Metrics
    graph.add_edge("total_loss", "opt", dst_port="loss")
    graph.add_edge("total_loss", "metrics", dst_port="loss")
    graph.add_edge("opt", "metrics", dst_port="opt_stats")
    
    # Add individual losses to metrics for visibility
    graph.add_edge("surr_loss", "metrics", dst_port="surr_loss")
    graph.add_edge("val_loss", "metrics", dst_port="val_loss")
    graph.add_edge("entropy", "metrics", dst_port="entropy")
    
    return graph


def create_ppo_update_graph(config: PPOConfig) -> Graph:
    """Create the full PPO update graph with loops."""
    graph = Graph()

    # 1. External Inputs (next state for GAE)
    graph.add_node("next_state", NODE_TYPE_SOURCE)

    # 2. Sample Batch
    graph.add_node("sample", "SampleBatch", params={"buffer_id": config.buffer_id})

    # 3. Compute GAE
    graph.add_node(
        "gae", 
        "AdvantageEstimation", 
        params={
            "method": "gae",
            "gamma": config.gamma, 
            "gae_lambda": config.gae_lambda,
            "num_envs": config.num_envs
        }
    )

    # 4. Epoch Loop
    # The body of the epoch loop is a minibatch loop
    train_step_graph = create_train_graph(config)

    epoch_body = Graph()
    epoch_body.add_node("batch", NODE_TYPE_SOURCE)
    epoch_body.add_node("advantages", NODE_TYPE_SOURCE)
    epoch_body.add_node("returns", NODE_TYPE_SOURCE)
    
    epoch_body.add_node(
        "mb_loop",
        "MinibatchIterator",
        params={
            "minibatch_size": config.minibatch_size,
            "body_graph": train_step_graph,
        },
    )
    epoch_body.add_edge("batch", "mb_loop", dst_port="batch")
    epoch_body.add_edge("advantages", "mb_loop", dst_port="advantages")
    epoch_body.add_edge("returns", "mb_loop", dst_port="returns")

    graph.add_node(
        "epoch_loop", "Loop", params={"iterations": config.epochs, "body_graph": epoch_body}
    )

    # Wiring
    graph.add_edge("sample", "gae", dst_port="batch")
    graph.add_edge("next_state", "gae", src_port="next_value", dst_port="next_value")
    graph.add_edge(
        "next_state", "gae", src_port="next_terminated", dst_port="next_terminated"
    )

    graph.add_edge("sample", "epoch_loop", dst_port="batch")
    graph.add_edge("gae", "epoch_loop", src_port="advantages", dst_port="advantages")
    graph.add_edge("gae", "epoch_loop", src_port="returns", dst_port="returns")

    return graph
