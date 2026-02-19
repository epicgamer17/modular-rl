import torch
import numpy as np
from replay_buffers.processors import MuZeroUnrollOutputProcessor


def test_reward_alignment():
    unroll_steps = 3
    n_step = 1
    gamma = 1.0
    num_players = 1

    # Simulate a game:
    # S0 -> S1 (Reward 10)
    # S1 -> S2 (Reward 20)
    # S2 -> S3 (Reward 30, Done)

    obs = torch.arange(4).float().view(-1, 1)  # S0, S1, S2, S3
    rewards = torch.tensor([10.0, 20.0, 30.0, 0.0])  # R1, R2, R3, Pad
    dones = torch.tensor([False, False, True, True])
    to_plays = torch.zeros(4).int()
    game_ids = torch.ones(4).int()

    buffers = {
        "observations": obs,
        "rewards": rewards,
        "values": torch.zeros_like(rewards),
        "to_plays": to_plays,
        "dones": dones,
        "game_ids": game_ids,
        "policies": torch.zeros((4, 1)),
        "actions": torch.zeros(4),
        "chances": torch.zeros((4, 1)),
        "legal_masks": torch.zeros((4, 1)).bool(),
        "ids": torch.zeros(4),
        "training_steps": torch.zeros(4),
    }

    processor = MuZeroUnrollOutputProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=gamma,
        num_actions=1,
        num_players=1,
        max_size=10,
    )

    batch = processor.process_batch([0], buffers)

    target_rewards = batch["rewards"][0].tolist()
    print(f"Input Rewards in Buffer: {rewards.tolist()}")
    print(f"Processed Target Rewards (Unroll 0 to 3): {target_rewards}")

    # MuZero Target Rewards Expected:
    # u=0 (root): 0.0
    # u=1 (first unroll): R1 = 10.0
    # u=2 (second unroll): R2 = 20.0
    # u=3 (third unroll): R3 = 30.0

    expected = [0.0, 10.0, 20.0, 30.0]
    assert target_rewards == expected, f"Expected {expected}, got {target_rewards}"
    print("Alignment Test Passed!")


if __name__ == "__main__":
    test_reward_alignment()
