import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import search

pytestmark = pytest.mark.unit

try:
    search.set_backend("cpp")
    HAS_CPP = True
    from search.search_cpp import ModularSearch, SearchConfig, ScoringConfig, SelectionConfig, BackpropConfig
except ImportError:
    HAS_CPP = False

@pytest.mark.skipif(not HAS_CPP, reason="C++ search backend not available.")
@pytest.mark.xfail(reason="search_cpp currently lacks internal Dirichlet noise injection in run_vectorized wrapper.")
def test_cpp_dirichlet_noise_injection_parity():
    """
    MATH: Parity test ensuring C++ backend injects Dirichlet noise
    consistent with AOS and Python backends.
    """
    # config.dirichlet_alpha = 0.3, fraction = 0.25
    # (Matches other backend tests)
    
    # Setup a search engine that SHOULD use Dirichlet
    # We need to build configs that enable it.
    # Note: SearchConfig in C++ currently DOES NOT have dirichlet fields.
    # This is part of the parity gap.
    
    config = MagicMock()
    config.num_simulations = 4
    config.search_batch_size = 1
    config.dirichlet_alpha = 0.3
    config.dirichlet_fraction = 0.25
    config.use_dirichlet = True
    config.known_bounds = None
    config.min_max_epsilon = 0.01
    config.discount_factor = 0.9
    config.num_players = 1
    config.backprop_method = "average"
    
    # Construct raw config objects for C++
    cpp_search_cfg = SearchConfig()
    cpp_search_cfg.num_actions = 2
    cpp_search_cfg.num_simulations = 4
    
    # If the bindings don't support it, this test is our way of documenting it.
    # Currently, run_vectorized in bindings.cpp ignores these fields because
    # they aren't even in the SearchConfig struct in C++.
    
    # mock_dirichlet.return_value = [0.1, 0.9]
    # Hand calc expecting: [0.625, 0.375]
    pass
