"""Batch iterators that decouple data sampling from the optimization loop.

The BlackboardEngine accepts any iterable of batch dicts. These iterators
wrap replay buffers to provide algorithm-specific sampling patterns:
- SingleBatchIterator: yields one batch (DQN, MuZero, Imitation)
- RepeatSampleIterator: yields N independent samples (training_iterations > 1)
- PPOEpochIterator: yields shuffled mini-batches over multiple epochs
"""

from typing import Any, Dict, Iterator

import torch

from data.storage.circular import ModularReplayBuffer


class SingleBatchIterator:
    """Yields exactly one batch sampled from the replay buffer."""

    def __init__(self, replay_buffer: ModularReplayBuffer, device: torch.device):
        self.replay_buffer = replay_buffer
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        batch = self.replay_buffer.sample()
        yield {
            k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()
        }


class RepeatSampleIterator:
    """Yields N independently sampled batches from the replay buffer.

    Used when config.training_iterations > 1 (e.g. Rainbow DQN).
    """

    def __init__(
        self,
        replay_buffer: ModularReplayBuffer,
        num_iterations: int,
        device: torch.device,
    ):
        self.replay_buffer = replay_buffer
        self.num_iterations = num_iterations
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _ in range(self.num_iterations):
            batch = self.replay_buffer.sample()
            yield {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }


class PPOEpochIterator:
    """Yields shuffled mini-batches over multiple epochs from a full rollout.

    Samples the entire buffer once, then iterates over it for multiple epochs,
    shuffling and slicing into mini-batches each epoch.
    """

    def __init__(
        self,
        replay_buffer: ModularReplayBuffer,
        num_epochs: int,
        num_minibatches: int,
        device: torch.device,
    ):
        self.replay_buffer = replay_buffer
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        batch = self.replay_buffer.sample()
        num_samples = batch["observations"].shape[0]
        minibatch_size = (
            num_samples + self.num_minibatches - 1
        ) // self.num_minibatches

        for _ in range(self.num_epochs):
            indices = torch.randperm(num_samples, device="cpu")
            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                batch_indices = indices[start:end]
                sub_batch = {
                    k: (
                        v[batch_indices].to(self.device)
                        if torch.is_tensor(v) and v.shape[0] == num_samples
                        else v
                    )
                    for k, v in batch.items()
                }
                yield sub_batch


# TODO: this feels somewhat hacky? should we have a buffer sampling component instead or something?
def infinite_ticks() -> Iterator[Dict[str, Any]]:
    """Yields empty dicts indefinitely for agents that don't need sampler data (e.g. inference only)."""
    while True:
        yield {}
