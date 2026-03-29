import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from agents.learner.base import UniversalLearner
from agents.learner.losses.value import ValueLoss
from agents.learner.losses.policy import PolicyLoss
from agents.learner.losses.loss_pipeline import LossPipeline
from agents.learner.losses.representations import ScalarRepresentation, ClassificationRepresentation

pytestmark = pytest.mark.unit

class SimpleNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.fc_v = nn.Linear(obs_dim, 1)
        self.fc_p = nn.Linear(obs_dim, num_actions)
        self.num_players = 1

    def learner_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = batch["observations"] # [B, T, D]
        # UniversalLearner expects [B, T, ...]
        B, T, D = obs.shape
        flat_obs = obs.reshape(B * T, D)
        pred_v = self.fc_v(flat_obs).reshape(B, T, 1)
        pred_p = self.fc_p(flat_obs).reshape(B, T, -1)
        return {"state_value": pred_v, "policy_logits": pred_p}

class MockTargetBuilder:
    def __init__(self, seed=42):
        self.seed = seed

    def build_targets(self, batch, predictions, network, current_targets):
        # Deterministic targets based on observations to ensure net1 and net2 get same targets
        obs = batch["observations"]
        B, T = obs.shape[:2]
        device = obs.device
        num_actions = network.fc_p.out_features
        
        # Use a deterministic function of observations for targets
        current_targets["values"] = obs.mean(dim=-1) 
        
        # Create a dummy policy target (uniform-ish but deterministic)
        policy_target = torch.ones(B, T, num_actions, device=device) / num_actions
        current_targets["policies"] = policy_target
        
        current_targets["value_mask"] = torch.ones(B, T, device=device)
        current_targets["policy_mask"] = torch.ones(B, T, device=device)
        current_targets["weights"] = torch.ones(B, device=device)
        current_targets["gradient_scales"] = torch.ones(B, T, device=device)

class MockConfig:
    def __init__(self, num_actions, minibatch_size, unroll_steps=0, num_players=1):
        self.minibatch_size = minibatch_size
        self.unroll_steps = unroll_steps
        self.game = type('GameConfig', (), {'num_actions': num_actions, 'num_players': num_players})()

def test_universal_learner_gradient_flow():
    """
    Ensures that loss.backward() actually computes gradients
    and optimizer.step() updates parameters.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_dim = 8
    num_actions = 4
    net = SimpleNetwork(obs_dim, num_actions)
    
    # Capture initial weights
    initial_v_weight = net.fc_v.weight.detach().clone()
    initial_p_weight = net.fc_p.weight.detach().clone()

    # 1. Setup Loss Pipeline
    v_loss = ValueLoss(
        device=device,
        representation=ScalarRepresentation(),
        loss_factor=1.0,
        optimizer_name="default"
    )
    p_loss = PolicyLoss(
        device=device,
        representation=ClassificationRepresentation(num_actions),
        loss_fn=torch.nn.functional.cross_entropy,
        loss_factor=1.0,
        optimizer_name="default"
    )
    
    config = MockConfig(num_actions=num_actions, minibatch_size=2, unroll_steps=2)
    pipeline = LossPipeline(config=config, modules=[v_loss, p_loss])
    
    # 2. Setup Learner with standard optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=10.0) # High LR to ensure visible change
    learner = UniversalLearner(
        agent_network=net,
        device=device,
        num_actions=num_actions,
        observation_dimensions=(obs_dim,),
        observation_dtype=torch.float32,
        target_builder=MockTargetBuilder(),
        loss_pipeline=pipeline,
        optimizer=optimizer,
        validator_params={"minibatch_size": 2, "unroll_steps": 2} # T=3
    )

    # 3. Dummy Batch [B, T, D]
    batch = {
        "observations": torch.randn(2, 3, obs_dim),
        "weights": torch.ones(2),
        "metrics": {}
    }

    # 4. Perform Update Step
    metrics_iter = learner.step([batch])
    next(metrics_iter)

    # 5. VERIFY WEIGHTS CHANGED
    # We don't check .grad here because UniversalLearner calls zero_grad() 
    # at the end of the step logic (before yield).
    assert not torch.allclose(net.fc_v.weight, initial_v_weight), "Value weights did not change after step"
    assert not torch.allclose(net.fc_p.weight, initial_p_weight), "Policy weights did not change after step"

def test_learner_multi_optimizer_routing():
    """
    Verifies that gradients are correctly isolated and routed when using 
    different optimizer groups for different network heads.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_dim = 8
    num_actions = 4
    net = SimpleNetwork(obs_dim, num_actions)
    
    # 1. Pipeline with separate optimizer targets
    v_loss = ValueLoss(
        device=device,
        representation=ScalarRepresentation(),
        loss_factor=1.0,
        optimizer_name="opt_v"
    )
    p_loss = PolicyLoss(
        device=device,
        representation=ClassificationRepresentation(num_actions),
        loss_fn=torch.nn.functional.cross_entropy,
        loss_factor=1.0,
        optimizer_name="opt_p"
    )
    
    config = MockConfig(num_actions=num_actions, minibatch_size=2, unroll_steps=2)
    pipeline = LossPipeline(config=config, modules=[v_loss, p_loss])
    
    # 2. Separate Optimizers
    opt_v = torch.optim.SGD(net.fc_v.parameters(), lr=10.0)
    opt_p = torch.optim.SGD(net.fc_p.parameters(), lr=10.0)
    
    learner = UniversalLearner(
        agent_network=net,
        device=device,
        num_actions=num_actions,
        observation_dimensions=(obs_dim,),
        observation_dtype=torch.float32,
        target_builder=MockTargetBuilder(),
        loss_pipeline=pipeline,
        optimizer={"opt_v": opt_v, "opt_p": opt_p},
        validator_params={"minibatch_size": 2, "unroll_steps": 2} # T=3
    )

    batch = {
        "observations": torch.randn(2, 3, obs_dim),
        "weights": torch.ones(2),
        "metrics": {}
    }

    # Step
    list(learner.step([batch]))

    # Capture state after both updated
    v_weight_after = net.fc_v.weight.detach().clone()
    p_weight_after = net.fc_p.weight.detach().clone()

    # Create a new batch
    batch2 = {
        "observations": torch.randn(2, 3, obs_dim),
        "weights": torch.ones(2),
        "metrics": {}
    }

    # Perform another step
    list(learner.step([batch2]))

    # Verify both changed again
    assert not torch.allclose(net.fc_v.weight, v_weight_after)
    assert not torch.allclose(net.fc_p.weight, p_weight_after)

from agents.learner.losses.reward import RewardLoss
from agents.learner.losses.to_play import ToPlayLoss

def test_muzero_comprehensive_gradient_flow():
    """
    Simulates a MuZero training step with Policy, Value, Reward, and ToPlay losses.
    Ensures all heads in a multi-head world model correctly receive gradients.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_dim = 8
    num_actions = 4
    num_players = 2
    
    class MultiHeadNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_v = nn.Linear(obs_dim, 1)
            self.fc_p = nn.Linear(obs_dim, num_actions)
            self.fc_r = nn.Linear(obs_dim, 1)
            self.fc_tp = nn.Linear(obs_dim, num_players)
            self.num_players = num_players

        def learner_inference(self, batch):
            obs = batch["observations"]
            B, T, D = obs.shape
            flat_obs = obs.reshape(B * T, D)
            return {
                "state_value": self.fc_v(flat_obs).reshape(B, T, 1),
                "policy_logits": self.fc_p(flat_obs).reshape(B, T, num_actions),
                "reward_logits": self.fc_r(flat_obs).reshape(B, T, 1),
                "to_play_logits": self.fc_tp(flat_obs).reshape(B, T, num_players)
            }

    net = MultiHeadNetwork()
    initial_weights = {k: p.detach().clone() for k, p in net.named_parameters()}

    # 1. Pipeline with all MuZero components
    modules = [
        ValueLoss(device, ScalarRepresentation()),
        PolicyLoss(device, ClassificationRepresentation(num_actions), torch.nn.functional.cross_entropy, 1.0),
        RewardLoss(device, ScalarRepresentation(), torch.nn.functional.mse_loss, 1.0),
        ToPlayLoss(device, ClassificationRepresentation(num_players), 1.0, mask_key="to_play_mask")
    ]
    
    config = MockConfig(num_actions=num_actions, minibatch_size=2, unroll_steps=2, num_players=num_players)
    pipeline = LossPipeline(config=config, modules=modules)

    class MuZeroTargetBuilder(MockTargetBuilder):
        def build_targets(self, batch, predictions, network, current_targets):
            super().build_targets(batch, predictions, network, current_targets)
            B, T = batch["observations"].shape[:2]
            current_targets["rewards"] = torch.randn(B, T)
            current_targets["to_plays"] = torch.randint(0, num_players, (B, T)).float()
            current_targets["reward_mask"] = torch.ones(B, T)
            current_targets["to_play_mask"] = torch.ones(B, T)

    learner = UniversalLearner(
        agent_network=net,
        device=device,
        num_actions=num_actions,
        observation_dimensions=(obs_dim,),
        observation_dtype=torch.float32,
        target_builder=MuZeroTargetBuilder(),
        loss_pipeline=pipeline,
        optimizer=torch.optim.SGD(net.parameters(), lr=10.0), # High LR
        validator_params={"minibatch_size": 2, "unroll_steps": 2, "num_players": num_players} # T=3
    )

    batch = {"observations": torch.randn(2, 3, obs_dim), "weights": torch.ones(2), "metrics": {}}
    list(learner.step([batch]))

    # Verify all layers updated
    for name, p in net.named_parameters():
        assert not torch.allclose(p, initial_weights[name]), f"MuZero head {name} did not update!"

def test_gradient_accumulation_correctness():
    """
    Ensures gradient accumulation simulates a larger batch size correctly
    by scaling the loss before backward.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_dim = 8
    num_actions = 4
    
    # Setup two identical networks
    net1 = SimpleNetwork(obs_dim, num_actions)
    net2 = SimpleNetwork(obs_dim, num_actions)
    net2.load_state_dict(net1.state_dict())

    # Build optimizer and learner for net1 (Standard, batch size 2)
    v_loss = ValueLoss(device=device, representation=ScalarRepresentation())
    config = MockConfig(num_actions=num_actions, minibatch_size=2, unroll_steps=2)
    pipeline1 = LossPipeline(config=config, modules=[v_loss])
    
    learner1 = UniversalLearner(
        agent_network=net1, 
        device=device, 
        num_actions=num_actions,
        observation_dimensions=(obs_dim,), 
        observation_dtype=torch.float32,
        target_builder=MockTargetBuilder(), 
        loss_pipeline=pipeline1,
        optimizer=torch.optim.SGD(net1.parameters(), lr=1.0),
        validator_params={"minibatch_size": 2, "unroll_steps": 2}
    )

    # Build optimizer and learner for net2 (Gradient Accumulation, 2 steps of size 1)
    config2 = MockConfig(num_actions=num_actions, minibatch_size=1, unroll_steps=2)
    pipeline2 = LossPipeline(config=config2, modules=[v_loss])
    learner2 = UniversalLearner(
        agent_network=net2, 
        device=device, 
        num_actions=num_actions,
        observation_dimensions=(obs_dim,), 
        observation_dtype=torch.float32,
        target_builder=MockTargetBuilder(), 
        loss_pipeline=pipeline2,
        optimizer=torch.optim.SGD(net2.parameters(), lr=1.0),
        gradient_accumulation_steps=2,
        validator_params={"minibatch_size": 1, "unroll_steps": 2}
    )

    # 1 large batch split into two small ones
    batch_full = {"observations": torch.randn(2, 3, obs_dim), "weights": torch.ones(2), "metrics": {}}
    batch_part1 = {"observations": batch_full["observations"][0:1], "weights": batch_full["weights"][0:1], "metrics": {}}
    batch_part2 = {"observations": batch_full["observations"][1:2], "weights": batch_full["weights"][1:2], "metrics": {}}

    # Standard step
    list(learner1.step([batch_full]))
    
    # Accumulation step
    list(learner2.step([batch_part1, batch_part2]))

    # Verified gradients/weights should be mathematically identical (ideally, with small float err)
    # Actually, SGD(B=2) == SGD(B=1) * 2 or similar? 
    # Loss is (L1 + L2) / 2 in standard. 
    # In Accumulation, we call L1/2.backward() then L2/2.backward().
    # Sum is (L1/2 + L2/2) = (L1 + L2)/2. YES!
    assert torch.allclose(net1.fc_v.weight, net2.fc_v.weight, atol=1e-5), "Weight update mismatch with gradient accumulation"
    assert torch.allclose(net1.fc_p.weight, net2.fc_p.weight, atol=1e-5), "Weight update mismatch with gradient accumulation"

def test_global_gradient_clipping():
    """
    Tier 1: Global Gradient Clipping test.
    Verifies that providing a small `clipnorm` or `max_grad_norm` actively restricts 
    the parameter updates compared to an unclipped optimization step.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    obs_dim = 8
    num_actions = 4
    
    # Setup two identical networks
    net_unclipped = SimpleNetwork(obs_dim, num_actions)
    net_clipped = SimpleNetwork(obs_dim, num_actions)
    net_clipped.load_state_dict(net_unclipped.state_dict())
    
    initial_v_weight = net_unclipped.fc_v.weight.detach().clone()
    
    # Unclipped Learner
    v_loss = ValueLoss(device=device, representation=ScalarRepresentation())
    config = MockConfig(num_actions=num_actions, minibatch_size=2, unroll_steps=2)
    pipeline1 = LossPipeline(config=config, modules=[v_loss])
    
    # We use a large learning rate so the unclipped step is massive 
    # and the clipped step is significantly restrained.
    learner_unclipped = UniversalLearner(
        agent_network=net_unclipped, 
        device=device, 
        num_actions=num_actions,
        observation_dimensions=(obs_dim,), 
        observation_dtype=torch.float32,
        target_builder=MockTargetBuilder(), 
        loss_pipeline=pipeline1,
        optimizer=torch.optim.SGD(net_unclipped.parameters(), lr=10.0),
        validator_params={"minibatch_size": 2, "unroll_steps": 2}
    )

    # Clipped Learner
    pipeline2 = LossPipeline(config=config, modules=[v_loss])
    # clipnorm is extremely small to forcefully trigger aggressive clipping
    learner_clipped = UniversalLearner(
        agent_network=net_clipped, 
        device=device, 
        num_actions=num_actions,
        observation_dimensions=(obs_dim,), 
        observation_dtype=torch.float32,
        target_builder=MockTargetBuilder(), 
        loss_pipeline=pipeline2,
        optimizer=torch.optim.SGD(net_clipped.parameters(), lr=10.0),
        clipnorm=0.001,
        validator_params={"minibatch_size": 2, "unroll_steps": 2}
    )

    batch = {"observations": torch.randn(2, 3, obs_dim), "weights": torch.ones(2), "metrics": {}}

    list(learner_unclipped.step([batch]))
    list(learner_clipped.step([batch]))

    # Calculate L2 norm of the update vectors
    unclipped_update_norm = torch.linalg.norm(net_unclipped.fc_v.weight - initial_v_weight)
    clipped_update_norm = torch.linalg.norm(net_clipped.fc_v.weight - initial_v_weight)

    # The update norm of the clipped network MUST be strictly smaller than the unclipped one
    assert clipped_update_norm < unclipped_update_norm, (
        f"Gradient clipping failed! "
        f"Clipped update norm: {clipped_update_norm}, Unclipped update norm: {unclipped_update_norm}"
    )
