import torch
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace
from agents.learner.base import UniversalLearner, StepResult
from agents.learner.callbacks import EarlyStopIteration

pytestmark = pytest.mark.unit


def test_universal_learner_multi_optimizer():
    config = SimpleNamespace(clipnorm=1.0)
    agent_network = MagicMock()

    # Create two dummy optimizers
    opt1 = MagicMock(spec=torch.optim.Optimizer)
    opt1.param_groups = [{"params": [torch.nn.Parameter(torch.randn(1))]}]
    opt2 = MagicMock(spec=torch.optim.Optimizer)
    opt2.param_groups = [{"params": [torch.nn.Parameter(torch.randn(1))]}]

    optimizers = {"opt1": opt1, "opt2": opt2}

    # Create two dummy schedulers
    sched1 = MagicMock()
    sched2 = MagicMock()
    schedulers = {"sched1": sched1, "sched2": sched2}

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        optimizer=optimizers,
        lr_scheduler=schedulers,
        clipnorm=config.clipnorm,
    )

    # Mock compute_step_result
    loss1 = torch.tensor(1.0, requires_grad=True)
    loss2 = torch.tensor(2.0, requires_grad=True)
    result = StepResult(
        loss={"opt1": loss1, "opt2": loss2}, loss_dict={"opt1": 1.0, "opt2": 2.0}
    )
    learner.compute_step_result = MagicMock(return_value=result)

    batch_iterator = [{"data": 1}]

    list(learner.step(batch_iterator))

    # Verify both optimizers were zeroed
    opt1.zero_grad.assert_called_once()
    opt2.zero_grad.assert_called_once()

    # Verify both optimizers were stepped
    opt1.step.assert_called_once()
    opt2.step.assert_called_once()

    # Verify both schedulers were stepped
    sched1.step.assert_called_once()
    sched2.step.assert_called_once()


def test_universal_learner_backward_compatibility():
    config = SimpleNamespace(clipnorm=0.0)
    opt = MagicMock(spec=torch.optim.Optimizer)

    learner = UniversalLearner(
        config=config,
        agent_network=MagicMock(),
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        optimizer=opt,
        clipnorm=config.clipnorm,
    )

    # Property access (removed in recent refactor, checking direct instead)
    assert "default" in learner.optimizers
    assert learner.optimizers["default"] is opt


def test_universal_learner_state_dict_multi_opt():
    config = SimpleNamespace(clipnorm=0.0)

    # Real parameters for state_dict
    p1 = torch.nn.Parameter(torch.randn(1))
    p2 = torch.nn.Parameter(torch.randn(1))

    opt1 = torch.optim.Adam([p1])
    opt2 = torch.optim.Adam([p2])

    learner = UniversalLearner(
        config=config,
        agent_network=torch.nn.Module(),  # Dummy
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        optimizer={"opt1": opt1, "opt2": opt2},
        clipnorm=config.clipnorm,
    )

    state = learner.state_dict()

    # Load into new learner
    learner2 = UniversalLearner(
        config=config,
        agent_network=torch.nn.Module(),
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        optimizer={"opt1": torch.optim.Adam([p1]), "opt2": torch.optim.Adam([p2])},
        clipnorm=config.clipnorm,
    )

    # Set different state to verify loading
    learner2.optimizers["opt1"].param_groups[0]["lr"] = 0.1

    learner2.load_state_dict(state)
    assert (
        learner2.optimizers["opt1"].param_groups[0]["lr"] == opt1.param_groups[0]["lr"]
    )


if __name__ == "__main__":
    # For quick manual run
    test_universal_learner_multi_optimizer()
    test_universal_learner_backward_compatibility()
    print("Self-tests passed!")
