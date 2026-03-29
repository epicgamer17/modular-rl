import pytest
import torch
from types import SimpleNamespace
from modules.utils import get_lr_scheduler
from agents.learner.base import UniversalLearner
from modules.models.agent_network import AgentNetwork
from utils.schedule import ScheduleConfig

pytestmark = pytest.mark.unit

def test_lr_scheduler_constant():
    """Tier 1: Verify constant LR scheduler behavior (Default)."""
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.001)
    
    # SimpleNamespace for config
    config = SimpleNamespace(lr_schedule="constant")
    
    scheduler = get_lr_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.001)

def test_lr_scheduler_linear_default_decay():
    """Tier 1: Verify linear LR scheduler with default 0.1 decay factor."""
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.001)
    
    config = SimpleNamespace(lr_schedule="linear", training_steps=1000)
    # No lr_final_factor or ScheduleConfig.final provided, should default to 0.1
    
    scheduler = get_lr_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)
    assert scheduler.end_factor == pytest.approx(0.1)

def test_lr_scheduler_linear_annealing_to_zero():
    """Tier 1: [ANALYTICAL ORACLE] Verify linear LR annealing to 0.0."""
    initial_lr = 0.01
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
    
    # Using ScheduleConfig to specify annealing to 0.0
    config = SimpleNamespace(
        lr_schedule=ScheduleConfig(type="linear", initial=initial_lr, final=0.0, decay_steps=1000),
        training_steps=1000
    )
    
    scheduler = get_lr_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)
    assert scheduler.end_factor == pytest.approx(0.0)
    
    # Analytical Oracle: verify values at specific steps
    # Step 0: 0.01
    assert scheduler.get_last_lr()[0] == pytest.approx(0.01)
    
    # Step 500: 0.005 (halfway)
    for _ in range(500):
        optimizer.step() # Optimizer step (dummy)
        scheduler.step()
    
    # LinearLR: lr = lr_initial * (1 + (end_factor - 1) * progress)
    # factor = 1 + (-1) * (500/1000) = 1 - 0.5 = 0.5
    assert scheduler.get_last_lr()[0] == pytest.approx(0.005)
    
    # Step 1000: 0.0
    for _ in range(500):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)

def test_learner_integration_steps_scheduler(ppo_config, net_factory):
    """Tier 1: Verify UniversalLearner correctly steps the LR scheduler."""
    torch.manual_seed(42)
    
    # Create a real but lightweight config
    initial_lr = 0.001
    ppo_config.learning_rate = initial_lr
    ppo_config.lr_schedule = ScheduleConfig(type="linear", initial=initial_lr, final=0.0, decay_steps=10)
    ppo_config.training_steps = 10
    
    device = torch.device("cpu")
    
    # Manually create a functional AgentNetwork instead of using broken net_factory
    from modules.backbones.mlp import MLPBackbone
    from modules.heads.policy import PolicyHead
    from agents.learner.losses.representations import ClassificationRepresentation
    
    def backbone_fn(input_shape):
        return MLPBackbone(input_shape=input_shape, widths=[8])
    
    rep = ClassificationRepresentation(num_classes=2)
    def policy_head_fn(**kwargs):
        return PolicyHead(representation=rep, **kwargs)
    
    network = AgentNetwork(
        input_shape=(4,),
        num_actions=2,
        memory_core_fn=backbone_fn,
        head_fns={"policy": policy_head_fn}
    )
    
    optimizer = torch.optim.Adam(network.parameters(), lr=initial_lr)
    scheduler = get_lr_scheduler(optimizer, ppo_config)
    
    from agents.learner.base import UniversalLearner
    from agents.learner.losses import LossPipeline
    from unittest.mock import MagicMock
    
    # Mock LossPipeline correctly
    mock_loss_pipeline = MagicMock(spec=LossPipeline)
    mock_loss_pipeline.run.return_value = (
        {"default": torch.tensor(1.0, requires_grad=True)},
        {"loss": 1.0},
        None
    )
    
    learner = UniversalLearner(
        agent_network=network,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        optimizer={"default": optimizer},
        lr_scheduler={"default": scheduler},
        loss_pipeline=mock_loss_pipeline,
        validator_params={"minibatch_size": 1, "unroll_steps": 0}
    )
    
    # Create a dummy batch iterator
    dummy_batch = {
        "obs": torch.randn(1, 1, 4),
        "actions": torch.zeros(1, 1, dtype=torch.long),
        "rewards": torch.zeros(1, 1),
        "dones": torch.zeros(1, 1),
        "values": torch.zeros(1, 1),
        "returns": torch.zeros(1, 1),
        "advantages": torch.zeros(1, 1),
        "log_prob": torch.zeros(1, 1),
    }
    
    # Need to mock learner_inference to return something with B, T
    network.learner_inference = MagicMock(return_value={
        "policy_logits": torch.randn(1, 1, 2)
    })
    
    # Run 5 steps
    iterator = [dummy_batch] * 5
    list(learner.step(iterator))
    
    # Expect LR to be halfway through decay (1.0 -> 0.0 over 10 steps, so 0.5 after 5 steps)
    # Progress = 5/10 = 0.5. Factor = 1 + (0 - 1) * 0.5 = 0.5. LR = 0.001 * 0.5 = 0.0005
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
