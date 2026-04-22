from typing import Dict, List, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from data.utils import discounted_cumulative_sums
from utils.utils import legal_moves_mask
from logging import warning



class OutputProcessor(ABC):
    """
    Processes indices indices retrieved from the Sampler into a final batch.
    """

    @abstractmethod
    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        """
        Args:
            indices: List of indices selected by the Sampler.
            buffers: A dictionary reference to the ReplayBuffer's internal storage
                     (e.g., {'obs': self.observation_buffer, 'rew': self.reward_buffer}).
        Returns:
            batch: A dictionary containing the final tensors for training.
        """
        pass  # pragma: no cover

    def clear(self):
        pass

class StackedOutputProcessor(OutputProcessor):
    """
    Chains multiple OutputProcessors.
    Each processor updates the 'batch' dictionary.
    """

    def __init__(self, processors: List[OutputProcessor]):
        self.processors = processors

    def process_batch(
        self,
        indices: List[int],
        buffers: Dict[str, torch.Tensor],
        batch: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        if batch is None:
            batch = {}

        for p in self.processors:
            # Processors should return a dict of new/updated keys
            # They receive the 'batch' so far to allow transformation (e.g. normalization)
            result = p.process_batch(indices, buffers, batch=batch, **kwargs)
            if result:
                batch.update(result)

        return batch

    def clear(self):
        for p in self.processors:
            p.clear()

class StandardOutputProcessor(OutputProcessor):
    """Returns data indices directly."""

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        return {key: buf[indices] for key, buf in buffers.items()}

class AdvantageNormalizer(OutputProcessor):
    """
    Normalizes advantages and formats batches for policy gradient methods.
    """

    def process_batch(
        self, indices: List[int], buffers: Dict[str, torch.Tensor], **kwargs
    ):
        # In PPO we usually sample the whole filled rollout and then minibatch in the learner.
        sl = slice(None) if indices is None else indices

        advantages = buffers["advantages"][sl].to(torch.float32)
        advantage_mean = advantages.mean()
        advantage_std = advantages.std()
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-10)

        return dict(
            observations=buffers["observations"][sl],
            actions=buffers["actions"][sl],
            rewards=buffers["rewards"][sl],
            dones=buffers["done"][sl],
            advantages=normalized_advantages,
            returns=buffers["returns"][sl],
            values=buffers["values"][sl],
            old_log_probs=buffers["old_log_probs"][sl],
            legal_moves_masks=buffers["legal_moves_masks"][sl],
        )