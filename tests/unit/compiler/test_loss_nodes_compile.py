import pytest
from core.graph import Graph
from core.graph import EdgeType
from agents.ppo.specs import register_ppo_specs
from agents.dqn.specs import register_dqn_specs
from runtime.bootstrap import bootstrap_runtime
from runtime.registry import clear_registry
from compiler.pipeline import compile_graph

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    bootstrap_runtime()
    register_ppo_specs()
    register_dqn_specs()


def test_ppo_loss_nodes_compile():
    """Verifies that the expanded PPO loss graph compiles."""
    g = Graph()

    # 1. Forward nodes
    g.add_node("obs", "Source")
    g.add_node("policy", "PolicyForward", params={"model_handle": "ppo_net"})
    g.add_edge("obs", "policy", dst_port="obs")

    # 2. Ratio
    g.add_node("old_log_probs", "Source")
    g.add_node("ratio", "Ratio")
    g.add_edge("policy", "ratio", src_port="log_prob", dst_port="new_log_probs")
    g.add_edge("old_log_probs", "ratio", dst_port="old_log_probs")

    # 3. Clip
    g.add_node("clip", "Clip", params={"low": 0.8, "high": 1.2})
    g.add_edge("ratio", "clip", dst_port="x")

    # 4. Surrogate
    g.add_node("adv", "Source")
    g.add_node("surr", "SurrogateMin")
    g.add_edge("ratio", "surr", dst_port="ratio")
    g.add_edge("clip", "surr", dst_port="clipped_ratio")
    g.add_edge("adv", "surr", dst_port="advantages")

    # 5. Value Loss
    g.add_node("returns", "Source")
    g.add_node("v_loss", "ValueLoss")
    g.add_edge("policy", "v_loss", src_port="values", dst_port="values")
    g.add_edge("returns", "v_loss", dst_port="returns")

    # 6. Mean and Weighted Sum
    g.add_node("mean_surr", "Mean")
    g.add_node("mean_v", "Mean")
    g.add_node("total_loss", "WeightedSum", params={"weights": {"surr": 1.0, "v": 0.5}})

    g.add_edge("surr", "mean_surr", dst_port="input")
    g.add_edge("v_loss", "mean_v", dst_port="input")
    g.add_edge("mean_surr", "total_loss", dst_port="surr")
    g.add_edge("mean_v", "total_loss", dst_port="v")

    # 7. Optimizer
    g.add_node("opt", "PPO_Optimizer", params={"model_handle": "ppo_net"})
    g.add_edge("total_loss", "opt", dst_port="loss")
    g.add_node("backward_total_loss", "Backward", params={"model_handle": "ppo_net"})
    g.add_edge("total_loss", "backward_total_loss", dst_port="loss")
    g.add_edge("backward_total_loss", "opt", edge_type=EdgeType.CONTROL)

    # This should compile without errors
    compiled = compile_graph(
        g,
        context="learner",
        model_handles={"ppo_net"},
        autodiff_lowering=False,
    )
    assert "ppo_net" in compiled.parameters


def test_dqn_loss_nodes_compile():
    """Verifies that the expanded DQN loss graph compiles."""
    g = Graph()

    # 1. Forward nodes
    g.add_node("obs", "Source")
    g.add_node("q_forward", "QForward", params={"model_handle": "online_q"})
    g.add_edge("obs", "q_forward", dst_port="obs")

    # 2. Gather Q
    g.add_node("actions", "Source")
    g.add_node("gather", "GatherActionQ")
    g.add_edge("q_forward", "gather", src_port="q_values", dst_port="q_values")
    g.add_edge("actions", "gather", dst_port="actions")

    # 3. Bellman Target
    g.add_node("next_obs", "Source")
    g.add_node("q_next", "QForward", params={"model_handle": "target_q"})
    g.add_edge("next_obs", "q_next", dst_port="obs")

    g.add_node("rewards", "Source")
    g.add_node("dones", "Source")
    g.add_node("bellman", "BellmanTarget", params={"gamma": 0.99})
    g.add_edge("q_next", "bellman", src_port="q_values", dst_port="next_q_values")
    g.add_edge("rewards", "bellman", dst_port="rewards")
    g.add_edge("dones", "bellman", dst_port="dones")

    # 4. MSE Loss
    g.add_node("mse", "MSELoss")
    g.add_edge("gather", "mse", src_port="q_selected", dst_port="pred")
    g.add_edge("bellman", "mse", src_port="target", dst_port="target")

    # 5. Optimizer
    g.add_node("opt", "Optimizer", params={"model_handle": "online_q"})
    g.add_edge("mse", "opt", dst_port="loss")

    compiled = compile_graph(
        g, context="learner", model_handles={"online_q", "target_q"}
    )
    assert "online_q" in compiled.parameters
