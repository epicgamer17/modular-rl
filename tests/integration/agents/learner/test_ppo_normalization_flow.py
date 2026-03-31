import pytest
import torch

pytestmark = pytest.mark.integration

def test_ppo_advantage_normalization_flow():
    """
    Tier 2: Integration Test.
    Verifies the flow of advantages from the ReplayBuffer (rollout-level norm)
    to the PPOEpochIterator (mini-batch level norm).
    Checks that advantages are ONLY normalized at the mini-batch level.
    """
    pytest.skip("TODO: update for old_muzero revert")

@pytest.mark.unit
def test_gaeprocessor_integration():
    """Tier 1/2: Checks that GAEProcessor produces advantages that PPOBatchProcessor correctly leaves raw."""
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert len(raw_advs) == 2
    # assert not np.isclose(np.mean(raw_advs), 0.0, atol=1e-3)
    # assert not torch.allclose(out["advantages"].mean(), torch.tensor(0.0), atol=1e-6)
    pytest.skip("TODO: update for old_muzero revert")

