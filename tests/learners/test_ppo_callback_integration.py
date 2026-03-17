import pytest
import torch
import torch.nn as nn
from agents.learners.base import UniversalLearner
from agents.learners.callbacks import PPOEarlyStoppingCallback, EarlyStopIteration
from losses.losses import LossPipeline, PPOPolicyLoss
from modules.agent_nets.modular import ModularAgentNetwork

pytestmark = pytest.mark.unit

class MockPolicyHead(nn.Module):
    def __init__(self, num_actions=2):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.num_actions = num_actions
    def forward(self, x):
        return torch.zeros((*x.shape[:-1], self.num_actions), device=x.device)

def test_ppo_kl_propagation_to_callback(ppo_config):
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    # 1. Setup Network and Heads
    agent_network = ModularAgentNetwork(
        config=ppo_config,
        input_shape=(4,),
        num_actions=2,
    )
    
    # 2. Setup Loss Pipeline
    ppo_loss = PPOPolicyLoss(
        config=ppo_config,
        device=device,
        clip_param=0.2,
        entropy_coefficient=0.01,
        optimizer_name="default"
    )
    pipeline = LossPipeline([ppo_loss])
    
    # 3. Setup Callback
    callback = PPOEarlyStoppingCallback(target_kl=0.01)
    
    # 4. Setup Learner
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=1e-3)
    learner = UniversalLearner(
        config=ppo_config,
        agent_network=agent_network,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        loss_pipeline=pipeline,
        optimizer=optimizer,
        callbacks=[callback]
    )
    
    # 5. Create a batch that will generate some KL
    # approx_kl = (old_log_probs - log_probs).mean()
    batch = {
        "observations": torch.randn(8, 4),
        "actions": torch.zeros(8, dtype=torch.long),
        "old_log_probs": torch.ones(8) * 0.5, # High old log probs
        "advantages": torch.ones(8),
    }
    
    # we need to mock the learner_inference to return a dict with "policies"
    def mock_learner_inference(batch):
        return {"policies": torch.zeros(8, 2)}
    agent_network.learner_inference = mock_learner_inference

    # 6. Run a step and check context/loss_dict
    # We call compute_step_result directly to verify propagation
    result = learner.compute_step_result(batch)
    
    assert "approx_kl" in result.loss_dict, "approx_kl should be in loss_dict after propagation"
    assert isinstance(result.loss_dict["approx_kl"], float)
    
    # 7. Verify Callback handles it
    # If KL > 1.5 * target_kl, it should raise EarlyStopIteration
    # Our dummy setup: old_log_probs = 0.5, log_probs = log(0.5) = -0.693
    # approx_kl = 0.5 - (-0.693) = 1.193 (approx)
    # 1.193 > 1.5 * 0.01 (0.015) -> Should raise
    
    with pytest.raises(EarlyStopIteration):
        learner.callbacks.on_backward_end(learner, result)

def test_ppo_callback_fail_fast():
    # Test valid target_kl
    PPOEarlyStoppingCallback(target_kl=0.01)
    
    # Test invalid target_kl
    with pytest.raises(AssertionError, match="target_kl must be positive"):
        PPOEarlyStoppingCallback(target_kl=-1.0)
        
    # Test missing key in loss_dict
    callback = PPOEarlyStoppingCallback(target_kl=0.01)
    from agents.learners.base import StepResult
    result = StepResult(loss={}, loss_dict={}) # Missing approx_kl
    
    with pytest.raises(AssertionError, match="missing from StepResult.loss_dict"):
        callback.on_backward_end(None, result)

if __name__ == "__main__":
    from configs.games.cartpole import CartPoleConfig
    from configs.agents.ppo import PPOConfig
    
    # Mock ppo_config fixture
    base_dict = {
        "steps_per_epoch": 2,
        "clip_param": 0.2,
        "entropy_coefficient": 0.01,
        "critic_coefficient": 0.5,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 1e-3,
        "adam_epsilon": 1e-8,
        "num_minibatches": 1,
        "actor_config": {"optimizer": torch.optim.Adam, "learning_rate": 1e-3, "adam_epsilon": 1e-8, "clipnorm": 0},
        "critic_config": {"optimizer": torch.optim.Adam, "learning_rate": 1e-3, "adam_epsilon": 1e-8, "clipnorm": 0},
        "action_selector": {"base": {"type": "categorical"}, "decorators": [{"type": "ppo_injector"}]},
        "policy_head": {"output_strategy": {"type": "categorical"}, "neck": {"type": "identity"}},
        "value_head": {"output_strategy": {"type": "scalar"}, "neck": {"type": "identity"}},
        "architecture": {"backbone": {"type": "identity"}},
    }
    game_cfg = CartPoleConfig()
    ppo_cfg = PPOConfig(base_dict, game_cfg)
    
    print("Running test_ppo_kl_propagation_to_callback...")
    test_ppo_kl_propagation_to_callback(ppo_cfg)
    print("test_ppo_kl_propagation_to_callback PASSED")
    
    print("Running test_ppo_callback_fail_fast...")
    test_ppo_callback_fail_fast()
    print("test_ppo_callback_fail_fast PASSED")
    
    print("All tests PASSED!")
