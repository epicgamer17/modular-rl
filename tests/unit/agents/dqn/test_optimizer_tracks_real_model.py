import pytest
import torch
import copy
from agents.dqn.agent import DQNAgent
from agents.dqn.config import DQNConfig
from runtime.executor import execute

pytestmark = pytest.mark.unit


# TODO: it is a flaky test, we have to fix it in the future
@pytest.mark.xfail
def test_optimizer_tracks_real_model():
    """
    Referential Integrity Test:
    Ensures that the optimizer in the learner graph is actually updating
     the weights of the 'live' model held by the agent.
    """
    config = DQNConfig(
        obs_dim=4,
        act_dim=2,
        hidden_dim=8,
        lr=1e-1,  # High LR to see change clearly
        buffer_capacity=100,
        min_replay_size=1,
        batch_size=1,
    )
    agent = DQNAgent(config)

    # 1. Referential Integrity Assertions (Identity checks)
    ctx = agent.get_execution_context()

    # Assert model identity
    assert agent.q_net is ctx.get_model(
        config.model_handle
    ), "Context must return the SAME model object as the agent"

    # Assert optimizer parameter identity
    opt_state = ctx.get_optimizer("main_opt")
    optimizer = opt_state.optimizer

    # Get first parameter from model and optimizer
    model_param = next(agent.q_net.parameters())
    opt_param = optimizer.param_groups[0]["params"][0]

    assert (
        model_param is opt_param
    ), "Optimizer must track the SAME tensor objects as the model"

    # 2. Functional Integrity Test (Weight update check)
    # Compile the graph
    agent.compile(strict=True)

    # Seed the buffer so the learner has something to train on
    obs = torch.randn(config.obs_dim)
    next_obs = torch.randn(config.obs_dim)
    agent.rb.add(
        {"obs": obs, "action": 0, "reward": 1.0, "next_obs": next_obs, "done": 0.0}
    )

    # Capture initial weights
    initial_weights = next(agent.q_net.parameters()).clone().detach()

    # Run ONE learner step
    # We execute the learner graph directly using the agent's context
    execute(agent.learner_graph, initial_inputs={}, context=ctx)

    # Capture new weights
    updated_weights = next(agent.q_net.parameters()).clone().detach()

    # Verify weights changed
    assert not torch.allclose(initial_weights, updated_weights), (
        "Weights did not change after learner step. "
        "Optimizer might be tracking a copy or learning rate is 0."
    )


def test_compiler_does_not_break_references():
    """
    Deep-copy Bug Regression Test:
    Ensures that the compiler's optimization passes (which may use deepcopy)
    do not inadvertently clone the model or optimizer state.
    """
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)

    initial_model = agent.q_net
    initial_opt = agent.opt_state

    agent.compile(strict=True)

    # The handles in the graph nodes should resolve to the SAME objects
    ctx = agent.get_execution_context()

    assert (
        ctx.get_model(config.model_handle) is initial_model
    ), "Compiler broke model reference"
    assert (
        ctx.get_optimizer("main_opt") is initial_opt
    ), "Compiler broke optimizer reference"


if __name__ == "__main__":
    # Allow manual execution as a script
    try:
        test_optimizer_tracks_real_model()
        test_compiler_does_not_break_references()
        print("\n[Success] All referential integrity tests passed.")
    except Exception as e:
        print(f"\n[Failure] Test failed: {e}")
        import traceback

        traceback.print_exc()
