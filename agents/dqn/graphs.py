from core.graph import (
    Graph,
    EdgeType,
    NODE_TYPE_SOURCE,
    NODE_TYPE_ACTOR,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_EXPLORATION,
    NODE_TYPE_METRICS_SINK,
)
from core.schema import TAG_OFF_POLICY, Schema, TensorSpec, Field
from agents.dqn.config import DQNConfig


def build_actor_graph(config: DQNConfig) -> Graph:
    """
    Builds the inference/actor graph for DQN.

    Inputs:
        obs: [obs_dim]
    Outputs:
        action: int
    """
    graph = Graph()

    # 1. Source for observations
    obs_spec = TensorSpec(shape=(config.obs_dim,), dtype="float32")
    graph.add_node(
        "obs_in", NODE_TYPE_SOURCE, schema_out=Schema([Field("obs", obs_spec)])
    )

    # 1b. Source for clock (to satisfy reachability)
    graph.add_node(
        "clock_in",
        NODE_TYPE_SOURCE,
        schema_out=Schema([Field("clock", TensorSpec(shape=(), dtype="int64"))]),
    )

    # 2. Q-Value computation (Single)
    q_spec = TensorSpec(shape=(config.act_dim,), dtype="float32")
    graph.add_node(
        "q_values",
        "QValuesSingle",
        params={"model_handle": config.model_handle},
        schema_out=Schema([Field("q_values", q_spec)]),
    )

    # 3. Epsilon Decay
    epsilon_spec = TensorSpec(shape=(), dtype="float32")
    graph.add_node(
        "epsilon_decay",
        "LinearDecay",
        params={
            "start_val": config.epsilon_start,
            "end_val": config.epsilon_end,
            "total_steps": config.epsilon_decay_steps,
        },
        schema_out=Schema([Field("epsilon", epsilon_spec)]),
    )
    graph.add_edge("clock_in", "epsilon_decay", dst_port="clock")

    # 4. Exploration (Epsilon-Greedy)
    action_spec = TensorSpec(shape=(), dtype="long")
    graph.add_node(
        "actor",
        "Exploration",
        params={"act_dim": config.act_dim},
        schema_out=Schema([Field("action", action_spec)]),
    )

    # Edges
    graph.add_edge("obs_in", "q_values", dst_port="obs")
    graph.add_edge("q_values", "actor", dst_port="q_values")
    graph.add_edge("epsilon_decay", "actor", dst_port="epsilon")

    return graph


def build_learner_graph(config: DQNConfig, collator) -> Graph:
    """
    Builds the training/learner graph for DQN.
    """
    graph = Graph()

    # 1. Replay Sampler
    # The output of the sampler is a batch dict following the collator's schema
    graph.add_node(
        "sampler",
        NODE_TYPE_REPLAY_QUERY,
        params={
            "buffer_id": config.buffer_id,
            "batch_size": config.batch_size,
            "min_size": config.min_replay_size,
        },
        schema_out=collator.schema,
    )

    # 2. TD Loss
    graph.add_node(
        "td_loss",
        "TDLoss",
        params={
            "model_handle": config.model_handle,
            "target_handle": config.target_handle,
            "gamma": config.gamma,
        },
    )

    # 3. Explicit gradient computation and application
    graph.add_node(
        "backward_td_loss",
        "Backward",
        params={
            "model_handle": config.model_handle,
            "optimizer_handle": "main_opt",
        },
    )
    graph.add_node(
        "accumulate_td_loss",
        "AccumulateGrad",
        params={
            "model_handle": config.model_handle,
            "k": 1,
        },
    )
    graph.add_node(
        "opt",
        "OptimizerStepEvery",
        params={
            "model_handle": config.model_handle,
            "optimizer_handle": "main_opt",
            "k": 1,
        },
    )

    # 4. Target Sync
    graph.add_node(
        "sync",
        NODE_TYPE_TARGET_SYNC,
        params={
            "model_handle": config.model_handle,
            "target_handle": config.target_handle,
            "sync_type": "periodic_hard",
            "sync_frequency": config.target_sync_frequency,
        },
    )

    # 5. Metrics computation (Average Q)
    graph.add_node("get_obs", "GetField", params={"field": "obs"})
    graph.add_node(
        "q_values_batch", "QForward", params={"model_handle": config.model_handle}
    )
    graph.add_node("avg_q", "ReduceMean")

    # 6. Metrics Sink
    graph.add_node(
        "metrics",
        NODE_TYPE_METRICS_SINK,
        params={"log_frequency": 10, "buffer_id": config.buffer_id},
    )

    # Wiring
    graph.add_edge("sampler", "td_loss", dst_port="batch")
    graph.add_edge("td_loss", "backward_td_loss", dst_port="loss")
    graph.add_edge("backward_td_loss", "accumulate_td_loss", edge_type=EdgeType.CONTROL)
    graph.add_edge("accumulate_td_loss", "opt", edge_type=EdgeType.CONTROL)
    # Note: sync is usually triggered externally or as a leaf in this graph
    graph.add_edge("opt", "sync")

    # metrics and other nodes
    graph.add_edge("sampler", "get_obs", dst_port="input")
    graph.add_edge("get_obs", "q_values_batch", dst_port="obs")
    graph.add_edge("q_values_batch", "avg_q", dst_port="input")
    graph.add_edge("avg_q", "metrics", dst_port="avg_q")
    graph.add_edge("td_loss", "metrics", dst_port="loss")

    return graph
