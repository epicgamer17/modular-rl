from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from torch import nn

from modules.world_models.inference_output import InferenceOutput


class BaseAgentNetwork(nn.Module, ABC):
    """
    Abstract Base Class for all Agent Networks (PPO, Rainbow, MuZero, etc.).
    Enforces a strict separation between Actor (Environment) and Learner (Training) APIs.
    """

    # ==========================================
    # 1. THE ACTOR'S API (Real World, t=0)
    # ==========================================
    @abstractmethod
    def obs_inference(self, obs: Any) -> InferenceOutput:
        """
        Called by the ActionSelector.
        Translates a real environment observation into semantic distributions and a root network_state.

        Args:
            obs: The current observation from the environment.

        Returns:
            InferenceOutput containing:
                - value: Expected value V(s)
                - policy: Action distribution (Categorical, Normal, etc.)
                - network_state: Opaque state for future search steps (if supported)
        """
        pass

    # ==========================================
    # 3. SEARCH API (Hypothetical World, t > 0)
    # ==========================================
    def hidden_state_inference(
        self,
        network_state: Dict[str, Any],
        action: torch.Tensor,
    ) -> InferenceOutput:
        """
        Translates (state, action) -> (next_state, reward, value, policy).
        Used by MCTS / Planning.
        """
        raise NotImplementedError(
            "hidden_state_inference not implemented for this AgentNetwork"
        )

    def afterstate_inference(
        self,
        network_state: Dict[str, Any],
        action: torch.Tensor,
    ) -> InferenceOutput:
        """
        Translates (state, action) -> (afterstate_value, policy_over_codes/outcomes).
        Used by Stochastic MCTS.
        """
        raise NotImplementedError(
            "afterstate_inference not implemented for this AgentNetwork"
        )

    def search_afterstate(
        self, network_state: Any, action: torch.Tensor
    ) -> InferenceOutput:
        """
        Called by Stochastic MCTS Algorithms.
        Steps the latent simulator to the intermediate 'afterstate' (post-action, pre-environment).

        Args:
            network_state: The opaque state from the previous step.
            action: The action taken.

        Returns:
            InferenceOutput containing afterstate predictions (e.g. chance logits).

        Raises:
            NotImplementedError: If the agent does not support stochastic afterstates.
        """
        raise NotImplementedError("This agent does not support stochastic afterstates.")

    # ==========================================
    # 3. THE LEARNER'S API (Historical Batch Optimization)
    # ==========================================
    @abstractmethod
    def learner_inference(self, batch: Any) -> Any:
        """
        Learner API: Pure logits for loss computation.

        Args:
            batch: Batch of data (observations, actions, etc.)

        Returns:
            Algorithm-specific learner predictions (typically a flat dict).
        """
        pass
