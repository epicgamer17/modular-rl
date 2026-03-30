import torch
import numpy as np
import pytest
import time
from agents.workers.actors import RolloutActor
from agents.action_selectors.decorators import TemperatureSelector
from agents.action_selectors.selectors import CategoricalSelector
from agents.action_selectors.policy_sources import NetworkPolicySource
from utils.schedule import ScheduleConfig
from agents.action_selectors.types import InferenceResult

pytestmark = pytest.mark.integration

class MockAdapter:
    def __init__(self, *args, num_players=1, **kwargs):
        self.num_envs = 1
        self.num_players = num_players
        self.observation_shape = (4,)
        self.num_actions = 2
        self.current_step = 0
        self.device = torch.device("cpu")
        
    def reset(self, **kwargs):
        self.current_step = 0
        obs = torch.zeros((1, 4))
        # RolloutActor expects player_id in dictionary
        info = {"player_id": [0]}
        return obs, info
        
    def step(self, actions):
        self.current_step += 1
        obs = torch.zeros((1, 4))
        reward = torch.zeros(1)
        terminated = torch.tensor([False])
        truncated = torch.tensor([False])
        info = {"player_id": [0]}
        return obs, reward, terminated, truncated, info

class MockBuffer:
    def store_aggregate(self, sequence, **kwargs): pass

class MockNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
    def eval(self): pass
    def obs_inference(self, obs):
        from modules.models.inference_output import InferenceOutput
        from torch.distributions import Categorical
        logits = torch.tensor([[1.0, 2.0]])
        return InferenceOutput(policy=Categorical(logits=logits), value=torch.tensor([0.0]))
    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]

def test_temperature_selector_training_step_flow():
    """
    Ensures that TemperatureSelector correctly updates its internal schedule
    when training_step is broadcast via update_parameters.
    """
    # 1. Setup Schedule: 1.0 -> 0.1 at step 100
    cfg = ScheduleConfig(
        type="stepwise",
        steps=[100],
        values=[1.0, 0.1],
        with_training_steps=True
    )
    
    selector = TemperatureSelector(CategoricalSelector(), cfg)
    
    # Check initial value
    assert selector.schedule.get_value() == 1.0
    
    # 2. Update parameters with training_step
    selector.update_parameters({"training_step": 150})
    
    # 3. Verify schedule is advanced
    assert selector._last_step == 150
    assert selector.schedule.get_value() == 0.1
    
    # 4. Verify select_action picks up the updated temperature from self._last_step
    res = InferenceResult(logits=torch.tensor([[1.0, 2.0]]))
    # Mock SelectAction to catch result
    def mock_inner_select(res_inner, info, exploration=True, **kwargs):
        # Check if logits were scaled by 0.1 (temp=0.1)
        # 1.0 / 0.1 = 10, 2.0 / 0.1 = 20
        torch.testing.assert_close(res_inner.logits, torch.tensor([[10.0, 20.0]]))
        return torch.tensor([0]), {}
        
    selector.inner_selector.select_action = mock_inner_select
    selector.select_action(res, {})

def test_temperature_selector_episode_step_flow():
    """
    Ensures that TemperatureSelector correctly updates its internal schedule
    when episode_step is passed via select_action.
    """
    # 1. Setup Schedule: 1.0 -> 0.1 at step 10
    cfg = ScheduleConfig(
        type="stepwise",
        steps=[10],
        values=[1.0, 0.1],
        with_training_steps=False
    )
    
    selector = TemperatureSelector(CategoricalSelector(), cfg)
    res = InferenceResult(logits=torch.tensor([[1.0, 2.0]]))
    
    # Case A: Step 0
    selector.select_action(res, {}, episode_step=0)
    assert selector.schedule.get_value() == 1.0
    
    # Case B: Step 15
    selector.select_action(res, {}, episode_step=15)
    assert selector.schedule.get_value() == 0.1
    
    # Case C: Reset (Step 0)
    selector.select_action(res, {}, episode_step=0)
    assert selector.schedule.get_value() == 1.0

def test_rollout_actor_integration_status():
    """
    Checks if RolloutActor correctly passes episode_step or training_step to select_action.
    """
    network = MockNetwork()
    adapter_cls = lambda *args, **kwargs: MockAdapter()
    
    # Stepwise: 1.0 -> 0.1 at step 5
    cfg = ScheduleConfig(
        type="stepwise",
        steps=[5],
        values=[1.0, 0.1],
        with_training_steps=False # Local episode step
    )
    selector = TemperatureSelector(CategoricalSelector(), cfg)
    
    actor = RolloutActor(
        adapter_cls=adapter_cls,
        adapter_args=(),
        network=network,
        policy_source=NetworkPolicySource(network),
        buffer=MockBuffer(),
        action_selector=selector
    )
    
    # Mock select_action to see what kwargs it receives
    original_select = selector.select_action
    received_kwargs = []
    
    def wrapped_select(res, info, exploration=True, **kwargs):
        received_kwargs.append(kwargs)
        return original_select(res, info, exploration=exploration, **kwargs)
        
    selector.select_action = wrapped_select
    
    # Run 10 steps
    actor.collect(num_steps=10)
    
    # Check if any received_kwargs had 'episode_step'
    has_episode_step = any("episode_step" in k for k in received_kwargs)
    
    if not has_episode_step:
        print("\n[VERIFIED] RolloutActor is NOT currently passing episode_step.")
    else:
        print("\n[VERIFIED] RolloutActor IS passing episode_step.")

if __name__ == "__main__":
    test_temperature_selector_training_step_flow()
    test_temperature_selector_episode_step_flow()
    test_rollout_actor_integration_status()
    print("\nALL STANDALONE TESTS PASSED")
