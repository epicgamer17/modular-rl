from replay_buffers.buffer_factories import create_dqn_buffer
from replay_buffers.transition import Transition, TransitionBatch
import torch
import numpy as np


def test_dqn_n_step_multi_process_fix():
    print("Testing DQN N-Step Multi-Process Fix (TransitionBatch loop)...")

    # Configuration for 3-step DQN
    dqn_config = type(
        "Config",
        (),
        {
            "n_step": 3,
            "discount_factor": 0.9,
            "per_alpha": 0.6,
            "per_beta_schedule": type("Beta", (), {"initial": 0.4}),
            "per_epsilon": 1e-6,
            "per_use_batch_weights": True,
            "per_use_initial_max_priority": True,
            "multi_process": False,  # Local test
        },
    )()

    observation_dimensions = (4,)
    num_actions = 2
    max_size = 100

    buffer = create_dqn_buffer(
        observation_dimensions,
        max_size,
        num_actions,
        batch_size=32,
        config=dqn_config,
        multi_process=False,
    )

    # 1. Create a batch of 5 transitions (partial trajectory)
    # We expect the first 2 steps to NOT be stored if n_step=3,
    # and the state to be preserved in the processor.
    ts = []
    for i in range(5):
        ts.append(
            Transition(
                observations=torch.ones(4) * i,
                actions=0,
                rewards=1.0,
                next_observations=torch.ones(4) * (i + 1),
                dones=False,
                terminated=False,
                truncated=False,
                legal_moves=[0, 1],
                next_legal_moves=[0, 1],
            )
        )

    batch = TransitionBatch(transitions=ts)

    print("Storing first batch (5 transitions)...")
    buffer.store_aggregate(batch)

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
    ts2 = []
    for i in range(5, 7):
        ts2.append(
            Transition(
                observations=torch.ones(4) * i,
                actions=0,
                rewards=1.0,
                next_observations=torch.ones(4) * (i + 1),
                dones=False,
                terminated=False,
                truncated=False,
                legal_moves=[0, 1],
                next_legal_moves=[0, 1],
            )
        )
    batch2 = TransitionBatch(transitions=ts2)
    print("Storing second batch (2 transitions)...")
    buffer.store_aggregate(batch2)

    # Now size should be 5
    print(f"Buffer size: {buffer.size}")
    assert buffer.size == 5, f"Expected size 5, got {buffer.size}"

    # 3. Store a terminal transition
    # This should trigger emission of tr5, tr6 AND the terminal transition
    ts3 = [
        Transition(
            observations=torch.ones(4) * 7,
            actions=0,
            rewards=1.0,
            next_observations=torch.ones(4) * 8,
            dones=True,
            terminated=True,
            truncated=False,
            legal_moves=[0, 1],
            next_legal_moves=[0, 1],
        )
    ]
    batch3 = TransitionBatch(transitions=ts3)
    print("Storing terminal batch (1 transition)...")
    buffer.store_aggregate(batch3)

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


if __name__ == "__main__":
    test_dqn_n_step_multi_process_fix()
