import torch
import pytest
from unittest.mock import MagicMock
from learner.pipeline.callbacks import ComponentCallbacks, Callback
from learner.core import Blackboard

pytestmark = pytest.mark.unit

def test_component_callbacks_hook_dispatch():
    """Verify ComponentCallbacks dispatches to the correct callback hook."""
    mock_callback = MagicMock(spec=Callback)
    component = ComponentCallbacks([mock_callback], hook="on_step_end")
    
    bb = Blackboard(batch={"data": 1})
    bb.predictions = {"p": 2}
    bb.targets = {"t": 3}
    bb.meta = {"loss": 0.5, "priorities": [1, 1]}
    
    component.execute(bb)
    
    # Check if on_step_end was called with correct unpacked data from blackboard
    mock_callback.on_step_end.assert_called_once()
    kwargs = mock_callback.on_step_end.call_args.kwargs
    
    assert kwargs["predictions"] == bb.predictions
    assert kwargs["targets"] == bb.targets
    assert kwargs["loss_dict"] == bb.meta # Current implementation passes meta as loss_dict
    assert kwargs["priorities"] == bb.meta["priorities"]
    assert kwargs["batch"] == bb.batch
    assert kwargs["meta"] == bb.meta

def test_component_callbacks_no_op_on_wrong_hook():
    """Verify ComponentCallbacks does nothing if the hook doesn't match."""
    # Note: Current implementation only supports on_step_end, but we check isolation
    mock_callback = MagicMock(spec=Callback)
    component = ComponentCallbacks([mock_callback], hook="unsupported_hook")
    
    bb = Blackboard(batch={})
    component.execute(bb)
    
    mock_callback.on_step_end.assert_not_called()
