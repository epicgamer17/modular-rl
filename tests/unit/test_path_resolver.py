import torch
import pytest
from core.blackboard import Blackboard
from core.path_resolver import resolve_blackboard_path

# Module-level marker strictly follows tests/README.md Tier 1 criteria.
pytestmark = pytest.mark.unit

def test_resolve_blackboard_path_contract():
    """
    Analytical Oracle & Contract Test for Blackboard Path Resolution.
    
    Contracts Verified:
    1. Fallback Search: Unqualified keys search targets -> data -> predictions.
    2. Auto-Normalization: 1D tensors [B] are automatically cast to [B, 1] for Learner compatibility.
    3. Explicit Routing: Dotted paths (e.g. 'data.x') bypass search and resolve directly.
    4. Nested Resolution: Supports deep nesting (e.g. 'meta.info.x').
    """
    blackboard = Blackboard()
    
    # 1. Setup Mock State with known values (Analytical Oracle targets)
    rewards_val = torch.tensor([1.5, -0.5])
    policies_val = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    predictions_val = torch.tensor([[10.0], [20.0]])
    
    blackboard.data["rewards"] = rewards_val
    blackboard.targets["policies"] = policies_val
    blackboard.predictions["values"] = predictions_val
    blackboard.meta["game_info"] = {"is_valid": True, "scores": torch.tensor([1, 2])}
    
    # --- 2. Verify Fallback Search Contract ---
    # Should find 'policies' in targets first.
    pol = resolve_blackboard_path(blackboard, "policies")
    assert pol.shape == (2, 2), f"Policies shape mismatch: expected (2, 2), got {pol.shape}"
    torch.testing.assert_close(pol, policies_val)
    
    # --- 3. Verify Auto-Normalization Contract ([B] -> [B, 1]) ---
    # 'rewards' is 1D in data. Resolution must add the time dimension (T=1).
    rew = resolve_blackboard_path(blackboard, "rewards")
    assert rew.ndim == 2, "Path resolver failed to add Time dimension to 1D tensor."
    assert rew.shape == (2, 1), f"Expected shape (2, 1), got {rew.shape}"
    torch.testing.assert_close(rew, rewards_val.unsqueeze(1))
    
    # --- 4. Verify Explicit Dotted Path Contract ---
    rew_explicit = resolve_blackboard_path(blackboard, "data.rewards")
    assert rew_explicit.shape == (2, 1)
    torch.testing.assert_close(rew_explicit, rewards_val.unsqueeze(1))
    
    # --- 5. Verify Metadata and Nested Resolution ---
    scores = resolve_blackboard_path(blackboard, "meta.game_info.scores")
    # ndim=1 -> unsqueeze(1) per resolver contract.
    assert scores.shape == (2, 1)
    torch.testing.assert_close(scores, torch.tensor([[1], [2]]))
    
    valid_flag = resolve_blackboard_path(blackboard, "meta.game_info.is_valid")
    assert valid_flag is True

    # --- 6. Verify Unhappy Paths (Fail-Fast) ---
    with pytest.raises(KeyError, match="Path root 'non' not found"):
        resolve_blackboard_path(blackboard, "non.existent")
    
    with pytest.raises(KeyError, match="Failed to resolve path 'data.missing'"):
        resolve_blackboard_path(blackboard, "data.missing")
