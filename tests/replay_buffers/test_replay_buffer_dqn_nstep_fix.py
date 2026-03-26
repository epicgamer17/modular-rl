import pytest
pytestmark = pytest.mark.integration

from agents.factories.replay_buffer import create_dqn_buffer
from replay_buffers.processors import NStepInputProcessor
import torch


def test_dqn_n_step_multi_process_fix(rainbow_cartpole_replay_config):
    print("Testing DQN N-Step Multi-Process Fix...")

    observation_dimensions = (4,)
    num_actions = 2
    max_size = 100

    buffer = create_dqn_buffer(
        observation_dimensions,
        max_size,
        num_actions,
        batch_size=32,
        config=rainbow_cartpole_replay_config,
    )

    def _store_step(step_idx, *, done=False, terminated=False):
        buffer.store(
            observations=torch.ones(4) * step_idx,
            actions=0,
            rewards=1.0,
            next_observations=torch.ones(4) * (step_idx + 1),
            dones=done,
            terminated=terminated,
            truncated=False,
            legal_moves=[0, 1],
            next_legal_moves=[0, 1],
        )

    # 1. Create a batch of 5 transitions (partial trajectory)
    # We expect the first 2 steps to NOT be stored if n_step=3,
    # and the state to be preserved in the processor.
    print("Storing first batch (5 transitions)...")
    for i in range(5):
        _store_step(i)

    # Expected behavior:
    # tr0 (i=0) needs tr1, tr2. Available at step 3.
    # tr1 (i=1) needs tr2, tr3. Available at step 4.
    # tr2 (i=2) needs tr3, tr4. Available at step 5.
    # tr3 (i=3) needs tr4, tr5 (not available).
    # so size should be 3.
    print(f"Buffer size: {buffer.size}")
    assert buffer.size == 3, f"Expected size 3, got {buffer.size}"

    # Check if reward is correct for tr0 (i=0)
    # R = 1.0 + 0.9*1.0 + 0.81*1.0 = 2.71
    rewards = buffer.buffers["rewards"][: buffer.size]
    print(f"Stored rewards: {rewards}")
    assert torch.allclose(
        rewards[0], torch.tensor(2.71)
    ), f"Expected 2.71, got {rewards[0]}"

    # 2. Store second batch (2 more transitions)
    # This should trigger emission of tr3 and tr4 from previous batch
    print("Storing second batch (2 transitions)...")
    for i in range(5, 7):
        _store_step(i)

    # Now size should be 5
    print(f"Buffer size: {buffer.size}")
    assert buffer.size == 5, f"Expected size 5, got {buffer.size}"

    # 3. Store a terminal transition
    # This triggers one emission immediately; then we flush remaining n-step
    # buffer entries to mirror end-of-trajectory finalization.
    print("Storing terminal batch (1 transition)...")
    _store_step(7, done=True, terminated=True)

    n_step_processor = buffer.input_processor.get_processor(NStepInputProcessor)
    assert n_step_processor is not None
    while n_step_processor.n_step_buffers[0]:
        processed = n_step_processor._emit_oldest(0)
        if processed is not None:
            buffer._store_processed(processed)
        n_step_processor.n_step_buffers[0].popleft()

    # Now size should be 8
    print(f"Buffer size: {buffer.size}")
    assert buffer.size == 8, f"Expected size 8, got {buffer.size}"

    # The last transition (index 7) should have reward 1.0 and done=True
    last_reward = buffer.buffers["rewards"][7]
    last_done = buffer.buffers["dones"][7]
    print(f"Last reward: {last_reward}, Done: {last_done}")
    assert last_reward == 1.0
    assert last_done == True

    print("Test Passed!")
