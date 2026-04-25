import pytest
import torch
from core.graph import Node
from runtime.context import ExecutionContext
from ops.math.schedule import op_linear_decay

pytestmark = pytest.mark.unit

def test_linear_decay_basic():
    """Verifies that LinearDecay correctly interpolates between start and end values over steps."""
    node = Node(
        node_id="decay",
        node_type="LinearDecay",
        params={"start_val": 1.0, "end_val": 0.0, "total_steps": 10}
    )
    
    # Step 0: Should be start_val
    ctx = ExecutionContext(env_step=0)
    val = op_linear_decay(node, {}, context=ctx)
    assert val == 1.0
    
    # Step 5: Should be 0.5 (halfway)
    ctx = ExecutionContext(env_step=5)
    val = op_linear_decay(node, {}, context=ctx)
    assert val == 0.5
    
    # Step 10: Should be end_val
    ctx = ExecutionContext(env_step=10)
    val = op_linear_decay(node, {}, context=ctx)
    assert val == 0.0
    
    # Step 100: Should remain at end_val (clamped)
    ctx = ExecutionContext(env_step=100)
    val = op_linear_decay(node, {}, context=ctx)
    assert val == 0.0

def test_linear_decay_input_clock():
    """Verifies that LinearDecay can use an explicit clock input instead of ExecutionContext."""
    node = Node(
        node_id="decay",
        node_type="LinearDecay",
        params={"start_val": 1.0, "end_val": 0.0, "total_steps": 10}
    )
    
    # Input clock = 2, should be 1.0 + (0.0 - 1.0) * (2/10) = 0.8
    val = op_linear_decay(node, {"clock": 2}, context=None)
    assert pytest.approx(val) == 0.8

def test_linear_decay_no_context():
    """Verifies that LinearDecay returns start_val if no context or input clock is provided."""
    node = Node(
        node_id="decay",
        node_type="LinearDecay",
        params={"start_val": 0.5, "end_val": 0.1, "total_steps": 100}
    )
    val = op_linear_decay(node, {}, context=None)
    assert val == 0.5
