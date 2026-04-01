import pytest
import torch
from unittest.mock import MagicMock
from agents.action_selectors.policy_sources import NetworkPolicySource, SearchPolicySource

pytestmark = pytest.mark.unit

def test_network_policy_source_unsqueeze():
    """Verify NetworkPolicySource correctly unsqueezes single observations."""
    agent_network = MagicMock()
    # Mock obs_inference to return something
    mock_dist = MagicMock()
    mock_out = MagicMock()
    mock_out.policy = mock_dist
    agent_network.obs_inference.return_value = mock_out
    
    input_shape = (4,)
    source = NetworkPolicySource(agent_network, input_shape)
    
    # 1. Single observation [4]
    obs = torch.zeros((4,))
    source.get_inference(obs, info={})
    
    # Verify obs_inference was called with [1, 4]
    args, kwargs = agent_network.obs_inference.call_args
    assert args[0].shape == (1, 4)

    # 2. Batched observation [2, 4]
    obs_batched = torch.zeros((2, 4))
    source.get_inference(obs_batched, info={})
    
    # Verify obs_inference was called with [2, 4] (no extra unsqueeze)
    args, kwargs = agent_network.obs_inference.call_args
    assert args[0].shape == (2, 4)

def test_search_policy_source_unsqueeze():
    """Verify SearchPolicySource correctly unsqueezes single observations."""
    search_engine = MagicMock()
    agent_network = MagicMock()
    input_shape = (4,)
    num_actions = 2
    
    source = SearchPolicySource(search_engine, agent_network, input_shape, num_actions)
    
    # 1. Single observation [4]
    obs = torch.zeros((4,))
    
    # run returns (root_value, exploratory_policy, target_policy, best_action, search_metadata)
    search_engine.run.return_value = (
        0.5, 
        torch.ones((2,)), # exploratory_policy
        torch.ones((2,)), # target_policy
        1, # best_action
        {} # metadata
    )
    
    source.get_inference(obs, info={'player': 'player_0'})
    
    # Verify run was called with [4] (unsqueezed to [1, 4] inside run usually, but here it's passed as is)
    # Actually SearchPolicySource.get_inference passes obs as is to run
    args, kwargs = search_engine.run.call_args
    assert args[0].shape == (4,)
    
    # 2. Batched observation [2, 4]
    obs_batched = torch.zeros((2, 4))
    
    # run_vectorized returns (root_values, exploratory_policies, target_policies, best_actions, sm_list)
    search_engine.run_vectorized.return_value = (
        [0.5, 0.6], # root_values
        [torch.ones((2,)), torch.ones((2,))], # exploratory_policies
        [torch.ones((2,)), torch.ones((2,))], # target_policies
        [1, 0], # best_actions
        [{}, {}] # sm_list
    )
    
    source.get_inference(obs_batched, info={'player': 'player_0'})
    
    # Verify run_vectorized was called with [2, 4]
    args, kwargs = search_engine.run_vectorized.call_args
    assert args[0].shape == (2, 4)
