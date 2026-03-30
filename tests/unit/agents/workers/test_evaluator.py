import torch
import torch.nn as nn
import numpy as np
import pytest
from agents.workers.actors import EvaluatorActor
from agents.workers.payloads import TaskRequest, TaskType
from agents.action_selectors.selectors import ArgmaxSelector
from agents.action_selectors.policy_sources import SearchPolicySource, NetworkPolicySource
from modules.models.agent_network import AgentNetwork
from agents.environments.adapters import BaseAdapter

pytestmark = pytest.mark.unit

class MockAdapter(BaseAdapter):
    def __init__(self, *args, num_players=1, episode_length=5, device=None, **kwargs):
        super().__init__(device=device or torch.device("cpu"))
        self.num_players = num_players
        self.episode_length = episode_length
        self.current_step = 0
        self.num_envs = 1
        self.observation_shape = (4,)
        self.agents = [f"player_{i}" for i in range(num_players)]
        
    def reset(self, **kwargs):
        self.current_step = 0
        obs = torch.zeros((1, 4))
        info = {"player_id": 0}
        return obs, info
        
    def step(self, actions):
        self.current_step += 1
        obs = torch.zeros((1, 4))
        # Constant reward of 1.0 for the acting player
        # In this mock, we'll just return a scalar reward for Gym, or handle all_rewards for multiplayer
        reward = torch.tensor([1.0])
        done = self.current_step >= self.episode_length
        terminated = torch.tensor([done])
        truncated = torch.tensor([False])
        
        # Calculate next player
        next_player = self.current_step % self.num_players
        info = {"player_id": next_player}
        
        # For multiplayer, add all_rewards
        if self.num_players > 1:
            all_rewards = torch.zeros(self.num_players)
            # The player who just acted gets the reward
            # previous player was (self.current_step - 1) % self.num_players
            prev_player = (self.current_step - 1) % self.num_players
            all_rewards[prev_player] = 1.0
            info["all_rewards"] = all_rewards
            
        return obs, reward, terminated, truncated, info

class MockAgent:
    def __init__(self, name="mock_agent"):
        self.name = name
    def obs_inference(self, obs, **kwargs):
        class DummyPolicy:
            def __init__(self):
                self.logits = torch.ones((1, 2))
                self.probs = torch.ones((1, 2)) / 2.0
        class Dummy:
            def __init__(self):
                self.policy = DummyPolicy()
                self.value = torch.tensor([0.0])
                self.extras = {}
        return Dummy()

def test_gym_evaluator_metrics():
    """Verify Simple Gym Evaluator returns min, max, mean."""
    # Setup mock network and adapter (1 player)
    network = nn.Module()
    network.obs_inference = lambda obs, **kwargs: MockAgent().obs_inference(obs, **kwargs)
    network.eval = lambda: None
    
    adapter_cls = lambda *args, **kwargs: MockAdapter(num_players=1, episode_length=10)
    
    evaluator = EvaluatorActor(
        adapter_cls=adapter_cls,
        adapter_args=(),
        network=network,
        policy_source=NetworkPolicySource(network),
        buffer=None,
        action_selector=ArgmaxSelector()
    )
    
    # Run evaluation for 3 episodes
    # MockAdapter gives 1.0 reward per step for 10 steps = 10.0 score per episode
    # We want to vary the score to test min/max/mean
    # Let's mock adapter.step to return different rewards based on episodes
    rewards = [5.0, 10.0, 15.0]
    ep_idx = 0
    original_step = evaluator.adapter.step
    def varied_step(actions):
        obs, reward, term, trunc, info = original_step(actions)
        if term[0]:
            nonlocal ep_idx
            # The total reward for the episode will be sum of rewards.
            # Our MockAdapter gives 1.0 per step. 
            # We'll just override the reward if we want specific totals easily.
            # Actually, let's just use the default and verify it gets 10.0
            pass
        return obs, reward, term, trunc, info
    
    evaluator.adapter.step = varied_step
    
    results = evaluator.evaluate(num_episodes=3)
    
    # Verify keys
    assert "mean_score" in results
    assert "min_score" in results
    assert "max_score" in results
    
    # Since all episodes were 10 steps of 1.0 reward
    assert results["mean_score"] == 10.0
    assert results["min_score"] == 10.0
    assert results["max_score"] == 10.0

def test_self_play_evaluator_metrics():
    """Verify Self Play returns mean score for P1."""
    # Setup 2 player mock environment
    adapter_cls = lambda *args, **kwargs: MockAdapter(num_players=2, episode_length=4)
    network = nn.Module()
    network.obs_inference = lambda obs, **kwargs: MockAgent().obs_inference(obs, **kwargs)
    network.eval = lambda: None

    evaluator = EvaluatorActor(
        adapter_cls=adapter_cls,
        adapter_args=(),
        network=network,
        policy_source=NetworkPolicySource(network),
        buffer=None,
        action_selector=ArgmaxSelector(),
        num_players=2
    )
    
    # P1 acts at step 0, 2. P2 at 1, 3.
    # Total steps = 4. 
    # Mock rewarded player gets 1.0. 
    # P1 gets 2.0, P2 gets 2.0.
    
    results = evaluator.evaluate(num_episodes=2)
    
    # Key should be mean_score (for p1)
    assert "mean_score" in results
    assert results["mean_score"] == 2.0

def test_vs_agent_evaluator_metrics():
    """Verify Test vs Agent cycles roles and reports p1/p2/mean."""
    adapter_cls = lambda *args, **kwargs: MockAdapter(num_players=2, episode_length=2)
    network = nn.Module()
    network.obs_inference = lambda obs, **kwargs: MockAgent().obs_inference(obs, **kwargs)
    network.eval = lambda: None
    
    # Test agent (Opponent)
    opponent = MockAgent(name="random")
    
    evaluator = EvaluatorActor(
        adapter_cls=adapter_cls,
        adapter_args=(),
        network=network,
        policy_source=NetworkPolicySource(network),
        buffer=None,
        action_selector=ArgmaxSelector(),
        num_players=2,
        test_agents=[opponent]
    )
    
    # Role cycling:
    # Round 1: Student is P1, Random is P2.
    # P1 acts step 0. Total score = 1.0 (step 0).
    # Round 2: Student is P2, Random is P1.
    # P2 acts step 1. Total score = 1.0 (step 1).
    
    results = evaluator.evaluate(num_episodes=1)
    
    # Structure: vs_random_score: {p1: 1.0, p2: 1.0, mean: 1.0}
    # And top level: p1_score, p2_score, mean_score
    
    assert "vs_random_score" in results
    assert results["vs_random_score"]["p1"] == 1.0
    assert results["vs_random_score"]["p2"] == 1.0
    
    assert "p1_score" in results
    assert "p2_score" in results
    assert "mean_score" in results
    
    assert results["p1_score"] == 1.0
    assert results["p2_score"] == 1.0
    assert results["mean_score"] == 1.0

if __name__ == "__main__":
    print("\n" + "="*50)
    print("RUNNING EVALUATOR TESTS (STANDALONE)")
    print("="*50)
    
    test_funcs = [
        test_gym_evaluator_metrics,
        test_self_play_evaluator_metrics,
        test_vs_agent_evaluator_metrics
    ]
    
    passed = 0
    for func in test_funcs:
        try:
            print(f"RUNNING: {func.__name__}...", end=" ", flush=True)
            func()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED\nERROR: {e}")
            import traceback
            traceback.print_exc()
            
    print("="*50)
    print(f"RESULTS: {passed}/{len(test_funcs)} PASSED")
    print("="*50 + "\n")
    
    if passed < len(test_funcs):
        import sys
        sys.exit(1)
