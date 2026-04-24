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
        "PPO_PolicyActor", 
        params={"model_handle": config.model_handle}, 
        tags=[TAG_ON_POLICY]
    )
    graph.add_edge("obs_in", "actor", dst_port="obs")
    return graph

def create_train_graph(config: PPOConfig) -> Graph:
    """Create the PPO training graph."""
    graph = Graph()
    
    # Source for the pre-processed minibatch (includes obs, action, advantages, returns)
    graph.add_node("traj_in", NODE_TYPE_SOURCE)
    
    # 1. PPO Objective
    graph.add_node(
        "ppo", 
        "PPO_Objective", 
        params={
            "model_handle": config.model_handle, 
            "clip_epsilon": config.clip_coef,
            "entropy_coef": config.ent_coef,
            "critic_coef": config.vf_coef
        }
    )
    
    # 2. Optimizer Step
    graph.add_node(
        "opt", 
        "PPO_Optimizer", 
        params={"optimizer_handle": config.optimizer_handle}
    )
    
    # 3. Metrics Sink
    graph.add_node(
        "metrics",
        NODE_TYPE_METRICS_SINK,
        params={"log_frequency": 32, "buffer_id": config.buffer_id},
    )

    # Wiring
    # Pass the minibatch as both the raw transitions and the GAE data
    graph.add_edge("traj_in", "ppo", dst_port="batch")
    graph.add_edge("traj_in", "ppo", dst_port="gae")
    
    graph.add_edge("ppo", "opt", dst_port="loss")
    graph.add_edge("ppo", "metrics", dst_port="loss")
    
    return graph
