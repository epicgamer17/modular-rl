import pytest
import torch
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

def test_priority_updater_callback_fails_fast():
    """Verify that PriorityUpdaterCallback now fails fast when update_priorities is missing."""
    class MockReplayBuffer:
        pass  # No update_priorities method
        
    class MockBetaSchedule:
        def get_value(self): return 0.4
        
    replay_buffer = MockReplayBuffer()
    callback = PriorityUpdaterCallback(replay_buffer, MockBetaSchedule())
    
    class MockLearner:
        pass
        
    batch = {"indices": [0], "ids": [0]}
    priorities = torch.tensor([1.0])
    
    with pytest.raises(AttributeError):
        callback.on_step_end(
            learner=MockLearner(),
            predictions={},
            targets={},
            loss_dict={},
            batch=batch,
            priorities=priorities
        )
