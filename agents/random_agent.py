import numpy as np


import torch
from torch.distributions import Categorical
from modules.models.inference_output import InferenceOutput

class RandomAgent:
    def __init__(self, num_actions: int = 1, name="random"):
        self.num_actions = num_actions
        self.name = name

    def obs_inference(self, obs: torch.Tensor, **kwargs) -> InferenceOutput:
        """
        Standardized inference interface for RandomAgent.
        Returns a uniform distribution over all actions.
        """
        batch_size = obs.shape[0]
        # Uniform probabilities across all actions
        probs = torch.ones((batch_size, self.num_actions), device=obs.device) / self.num_actions
        
        return InferenceOutput(
            policy=Categorical(probs=probs),
            # Random agent doesn't have a value function, but we return 0.0 for consistency
            value=torch.zeros((batch_size,), device=obs.device)
        )
