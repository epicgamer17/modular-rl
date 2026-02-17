import torch
from typing import Any, Dict, Tuple
from agents.policies.direct_policy import DirectPolicy


class PPOPolicy(DirectPolicy):
    """
    Extended DirectPolicy for PPO that returns action, log_prob, and value.
    """

    def compute_action(
        self, obs: Any, info: Dict[str, Any] = None, exploration: bool = False
    ) -> Any:
        """
        Computes an action given an observation and info.
        For PPO, we only use the actor output from the model.
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            # If obs_tensor matches input_shape, it's a single observation -> unsqueeze.
            if obs_tensor.dim() == len(self.model.policy.neck.input_shape):
                obs_tensor = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            # Get current policy distribution
            distribution = self.model.policy.get_distribution(obs_tensor)

        if exploration:
            action = distribution.sample()
        else:
            # Greedy action for PPO (mode of distribution)
            if isinstance(distribution, torch.distributions.Categorical):
                action = distribution.probs.argmax(dim=-1)
            else:
                action = distribution.mean

        if action.dim() > 0 and action.shape[0] == 1:
            action = action.squeeze(0)

        return action

    def compute_action_with_info(
        self, obs: Any, info: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Computes action, log probability, and value for PPO training.

        Returns:
            Tuple of (action, log_probability, value).
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            obs_tensor = obs.to(self.device)
            # We assume model.policy.neck exists if using modular heads
            if obs_tensor.dim() == len(self.model.policy.neck.input_shape):
                obs_tensor = obs_tensor.unsqueeze(0)

        with torch.inference_mode():
            # Get current policy distribution and critic value
            distribution = self.model.policy.get_distribution(obs_tensor)
            value = self.model.value(obs_tensor)

        # Sample action from distribution
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return (
            action.squeeze(),
            log_prob.squeeze().item(),
            value.squeeze().item(),
        )
