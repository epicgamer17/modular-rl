import torch
import time
import numpy as np
from replay_buffers.processors import NStepUnrollProcessor


def benchmark_n_step():
    # Setup realistic parameters
    batch_size = 256
    unroll_steps = 5
    n_step = 10
    gamma = 0.997
    num_actions = 18
    num_players = 2
    max_size = 1000

    proc = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=gamma,
        num_actions=num_actions,
        num_players=num_players,
        max_size=max_size,
    )

    # larger window for lookahead
    horizon = unroll_steps + n_step + 5

    # 1. Create Dummy Data
    raw_rewards = torch.randn((batch_size, horizon), device="cpu")
    raw_values = torch.randn((batch_size, horizon), device="cpu")
    raw_to_plays = torch.randint(
        0, num_players, (batch_size, horizon), dtype=torch.int16, device="cpu"
    )
    raw_dones = torch.zeros((batch_size, horizon), dtype=torch.bool, device="cpu")
    raw_truncated = torch.zeros((batch_size, horizon), dtype=torch.bool, device="cpu")
    valid_mask = torch.ones((batch_size, horizon), dtype=torch.bool, device="cpu")

    # Warmup
    print("Warming up...")
    for _ in range(5):
        proc._compute_n_step_targets(
            batch_size,
            raw_rewards,
            raw_values,
            raw_to_plays,
            raw_dones,
            raw_truncated,
            valid_mask,
            "cpu",
        )

    # Benchmarking
    iterations = 20
    start_time = time.time()
    for _ in range(iterations):
        proc._compute_n_step_targets(
            batch_size,
            raw_rewards,
            raw_values,
            raw_to_plays,
            raw_dones,
            raw_truncated,
            valid_mask,
            "cpu",
        )
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(
        f"Average execution time (batch={batch_size}, unroll={unroll_steps}, n={n_step}): {avg_time*1000:.2f} ms"
    )


if __name__ == "__main__":
    benchmark_n_step()
