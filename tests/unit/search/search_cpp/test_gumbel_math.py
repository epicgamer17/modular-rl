import math
import pytest
import torch

try:
    import search
    search.set_backend("cpp")
    from search_cpp import NodeArena, DecisionNode
    HAS_CPP = True
except (ImportError, RuntimeError):
    HAS_CPP = False

pytestmark = pytest.mark.unit

@pytest.mark.skipif(not HAS_CPP, reason="C++ search backend not available.")
@pytest.mark.xfail(reason="DecisionNode.get_v_mix() is not currently implemented in the C++ backend.")
def test_gumbel_v_mix_math_verification():
    """
    Verifies the Gumbel v_mix calculation in the C++ backend.
    """
    arena = NodeArena()
    root_idx = arena.create_decision(prior=1.0)
    root = arena.decision(root_idx)
    
    # Hand-calculated oracle: 3.09375
    # (Setup code same as Python/AOS would go here if C++ supported it)
    
    # Check if method exists
    assert hasattr(root, "get_v_mix"), "C++ DecisionNode is missing get_v_mix() method"
    vmix = root.get_v_mix()
    assert math.isclose(vmix, 3.09375, rel_tol=1e-7)
