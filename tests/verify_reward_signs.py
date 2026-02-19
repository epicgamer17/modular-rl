import torch
import numpy as np
from replay_buffers.processors import MuZeroUnrollOutputProcessor


def test_reward_flipping():
    unroll_steps = 2
    n_step = 2
    gamma = 1.0
    num_players = 2

    # Simulate a game where:
    # Step 0: Player 0 acts, reward 10, turn changes to Player 1
    # Step 1: Player 1 acts, reward 0, turn changes back to Player 0
    # Step 2: Player 0 acts, reward 20, turn changes to Player 1

    to_plays = [0, 1, 0, 1, 1]
    rewards = [10.0, 5.0, 20.0, 0.0, 0.0]
    values = [100.0, -100.0, 100.0, -100.0, 0.0]
    dones = [False, False, False, True, True]
    game_ids = [1, 1, 1, 1, 1]

    buffers = {
        "observations": torch.zeros((10, 1)),
        "rewards": torch.tensor(rewards).float(),
        "values": torch.tensor(values).float(),
        "to_plays": torch.tensor(to_plays).int(),
        "dones": torch.tensor(dones).bool(),
        "game_ids": torch.tensor(game_ids).int(),
        "policies": torch.zeros((10, 1)),
        "actions": torch.zeros((10,)),
        "chances": torch.zeros((10, 1)),
        "legal_masks": torch.zeros((10, 1)).bool(),
        "ids": torch.zeros((10,)),
        "training_steps": torch.zeros((10,)),
    }

    processor = MuZeroUnrollOutputProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=gamma,
        num_actions=1,
        num_players=num_players,
        max_size=10,
    )

    indices = [0]
    batch = processor.process_batch(indices, buffers)

    target_values = batch["values"][0]
    print(f"To Plays: {to_plays[:5]}")
    print(f"Rewards: {rewards[:5]}")
    print(f"Target Values: {target_values.tolist()}")

    # Trace for V0 (Root u=0):
    # current_player = P0
    # k=0: r_idx=0, step_player=P0, sign=1. Sum = 10.
    # k=1: r_idx=1, step_player=P1, sign=-1. Sum = 10 - 5 = 5.
    # boost: boot_idx=2, boot_player=P0, sign=1. Sum = 5 + 100 = 105.
    expected_v0 = 105.0
    assert (
        target_values[0].item() == expected_v0
    ), f"Expected {expected_v0}, got {target_values[0].item()}"
    print("Test passed!")


if __name__ == "__main__":
    test_reward_flipping()
