import pytest
import numpy as np
from replay_buffers.processors import GAEProcessor, NStepInputProcessor

pytestmark = pytest.mark.unit


def test_gae_processor_math():
    processor = GAEProcessor(gamma=0.99, gae_lambda=0.95)

    # Simulate a short trajectory
    transitions = [
        {"rewards": 1.0, "values": 0.5, "dones": False, "policies": 0.1},
        {"rewards": 2.0, "values": 1.5, "dones": False, "policies": 0.2},
        {"rewards": 0.0, "values": 0.0, "dones": True, "policies": 0.5},
    ]

    result = processor.process_sequence(None, transitions=transitions)

    assert "transitions" in result
    assert len(result["transitions"]) == 3
    # Verify the math properties were injected
    assert "advantages" in result["transitions"][0]
    assert "returns" in result["transitions"][0]
    assert "log_probabilities" in result["transitions"][0]


def test_nstep_processor_accumulation():
    # Test N-Step buffering and reward accumulation
    processor = NStepInputProcessor(n_step=2, gamma=0.9)

    t1 = {"player": 0, "rewards": 1.0, "dones": False, "next_observations": "obs1"}
    t2 = {"player": 0, "rewards": 2.0, "dones": False, "next_observations": "obs2"}
    t3 = {"player": 0, "rewards": 3.0, "dones": True, "next_observations": "obs3"}

    # Step 1: Buffer is filling, should return None
    res1 = processor.process_single(**t1)
    assert res1 is None

    # Step 2: Buffer hits n_step=2, should emit the modified first transition
    res2 = processor.process_single(**t2)
    assert res2 is not None
    # Reward should be t1.reward + gamma * t2.reward (1.0 + 0.9 * 2.0 = 2.8)
    assert np.isclose(res2["rewards"], 2.8)
    assert res2["next_observations"] == "obs2"

    # Step 3: Terminal state
    res3 = processor.process_single(**t3)
    assert res3 is not None
    assert np.isclose(res3["rewards"], 2.0 + 0.9 * 3.0)
