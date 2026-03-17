import pytest
import torch
from unittest.mock import MagicMock
from agents.learners.callbacks import ResetNoiseCallback, PriorityUpdaterCallback

pytestmark = pytest.mark.regression

def test_reset_noise_callback_fails_fast():
    """Verify that ResetNoiseCallback now fails fast when reset_noise is missing."""
    callback = ResetNoiseCallback()
    
    class MockNetwork:
        pass  # No reset_noise method
        
    class MockLearner:
        def __init__(self):
            self.agent_network = MockNetwork()
            
    learner = MockLearner()
    
    with pytest.raises(AttributeError):
        callback.on_optimizer_step_end(learner)

def test_priority_updater_callback_functional_fails_fast():
    """Verify that PriorityUpdaterCallback fails fast when the callable is invalid or missing dependencies."""
    # 1. Test missing priority_update_fn
    with pytest.raises(TypeError):
        PriorityUpdaterCallback() # Missing required positional arguments

    # 2. Test fail-fast on step end when kwargs are missing
    callback = PriorityUpdaterCallback(
        priority_update_fn=lambda idx, p, ids: None,
        set_beta_fn=lambda b: None,
        per_beta_schedule=MagicMock()
    )
    
    with pytest.raises(KeyError):
        # Should fail because 'batch' and 'priorities' are missing in kwargs
        callback.on_step_end(
            learner=None,
            predictions={},
            targets={},
            loss_dict={},
        )

def test_priority_updater_callback_beta_fails_fast():
    """Verify that PriorityUpdaterCallback fails fast when dependencies are missing."""
    with pytest.raises(TypeError):
        # Should fail because it misses required arguments
        callback = PriorityUpdaterCallback(priority_update_fn=lambda idx, p, ids: None)
