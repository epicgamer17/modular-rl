import pytest
import torch
from unittest.mock import MagicMock
from agents.action_selectors.policy_sources import SearchPolicySource
from modules.models.inference_output import InferenceOutput

pytestmark = pytest.mark.unit

def test_search_policy_source_uses_run_for_single_batch():
    """
    Ensures that SearchPolicySource calls search.run instead of run_vectorized
    when provided with a single observation, and that it correctly squeezes 
    the observation dimension.
    """
    mock_search = MagicMock()
    # Mock run to return the expected 5-tuple
    mock_search.run.return_value = (
        0.5,                              # root_v
        torch.ones(4) / 4,               # expl_p
        torch.ones(4) / 4,               # target_p
        0,                                # best_a
        {"meta": "data"}                  # sm
    )
    
    mock_network = MagicMock()
    source = SearchPolicySource(search_engine=mock_search, agent_network=mock_network)
    
    # Input with batch size 1
    obs = torch.randn(1, 3, 8, 8)
    info = [{"player": 0}]
    
    result = source.get_inference(obs, info)
    
    # Assert run was called, not run_vectorized
    mock_search.run.assert_called_once()
    mock_search.run_vectorized.assert_not_called()
    
    # Verify shape modification: obs[0] should be passed (squeezed)
    args, kwargs = mock_search.run.call_args
    passed_obs = args[0]
    assert passed_obs.shape == (3, 8, 8), f"Expected squeezed obs (3, 8, 8), got {passed_obs.shape}"
    
    # Verify info was unwrapped if it was a list
    passed_info = args[1]
    assert passed_info == {"player": 0}
    
    # Verify the return result is still 'batched' (lists of 1) for higher-level actors
    assert result.value.shape == (1, 1)
    assert result.probs.shape == (1, 4)

def test_search_policy_source_uses_run_vectorized_for_multi_batch():
    """
    Ensures that SearchPolicySource calls search.run_vectorized when 
    provided with multiple observations.
    """
    mock_search = MagicMock()
    # Mock run_vectorized to return the expected 5-tuple of lists
    mock_search.run_vectorized.return_value = (
        [0.5, 0.6],
        [torch.ones(4)/4, torch.ones(4)/4],
        [torch.ones(4)/4, torch.ones(4)/4],
        [0, 1],
        [{}, {}]
    )
    
    mock_network = MagicMock()
    source = SearchPolicySource(search_engine=mock_search, agent_network=mock_network)
    
    # Input with batch size 2
    obs = torch.randn(2, 3, 8, 8)
    info = [{"player": 0}, {"player": 1}]
    
    result = source.get_inference(obs, info)
    
    # Assert run_vectorized was called, not run
    mock_search.run_vectorized.assert_called_once()
    mock_search.run.assert_not_called()
    
    assert result.value.shape == (2, 1)

def test_modular_search_run_unsqueezes_obs():
    """
    Ensures that ModularSearch.run correctly unsqueezes an observation
    that has no batch dimension before passing it to AgentNetwork.obs_inference.
    """
    from search.search_py.modular_search import ModularSearch
    
    # Minimal config for ModularSearch
    mock_config = MagicMock()
    mock_config.gumbel = False
    mock_config.bootstrap_method = "value"
    mock_config.policy_extraction = "visit"
    mock_config.num_simulations = 1
    mock_config.search_batch_size = 0
    mock_config.max_search_depth = 5
    mock_config.known_bounds = None
    mock_config.min_max_epsilon = 1e-6
    mock_config.discount_factor = 0.99
    mock_config.pb_c_init = 1.25
    mock_config.pb_c_base = 19652
    mock_config.stochastic = False
    
    mock_network = MagicMock()
    mock_network.input_shape = (3, 8, 8)
    
    # Mock obs_inference to return a minimal valid InferenceOutput
    mock_policy = MagicMock()
    mock_policy.logits = torch.randn(1, 4)
    mock_output = InferenceOutput(
        recurrent_state={"dynamics": torch.randn(1, 16)},
        value=torch.tensor([0.5]),
        policy=mock_policy,
        reward=None,
        to_play=None
    )
    mock_network.obs_inference.return_value = mock_output
    
    search = ModularSearch(mock_config, device=torch.device("cpu"), num_actions=4)
    
    # Input with NO batch dimension [C, H, W]
    obs = torch.randn(3, 8, 8) 
    info = {"player": 0}
    
    # We catch exceptions because we only care about the first call to obs_inference
    # which is at the very beginning of the run() method.
    try:
        search.run(obs, info, mock_network)
    except Exception:
        pass
    
    # Verify that obs_inference was called with a batch dimension added
    mock_network.obs_inference.assert_called()
    args, _ = mock_network.obs_inference.call_args
    passed_obs = args[0]
    assert passed_obs.shape == (1, 3, 8, 8), f"ModularSearch.run should have unsqueezed (3, 8, 8) to (1, 3, 8, 8). Got {passed_obs.shape}"
