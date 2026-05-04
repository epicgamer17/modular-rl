import torch
import torch.nn.functional as F
from functools import partial
from typing import Tuple, Callable
import math


# TODO: at the moment algorithm specific in future try to fix, when needed. for now keep simple as is
def scalar_extractor(predictions: torch.Tensor) -> torch.Tensor:
    """
    Standard argmax over scalar Q-values.

    Args:
        predictions (torch.Tensor): The predictions from the model.

    Returns:
        torch.Tensor: The actions selected by the model.
    """
    return torch.argmax(predictions, dim=1, keepdim=True)


def categorical_extractor(
    predictions: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """
    Calculate expected Q-values from a distribution, then argmax.

    Args:
        predictions (torch.Tensor): The predictions from the model.
        support (torch.Tensor): The support for the distribution.

    Returns:
        torch.Tensor: The actions selected by the model.
    """
    probs = F.softmax(predictions, dim=-1)
    q_values = (probs * support).sum(dim=-1)
    return torch.argmax(q_values, dim=1, keepdim=True)


def standard_selector(
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    obs: torch.Tensor,
    extractor_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard DQN: Target network picks and evaluates.

    Args:
        model (nn.Module): The model to use for action selection.
        target_model (nn.Module): The target model to use for action selection.
        obs (torch.Tensor): The observations to use for action selection.
        extractor_fn (Callable): The function to use for action selection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the target predictions and the actions selected by the model.
    """
    # Skip target compute during actor rollout
    target_preds = target_model(obs) if target_model is not None else None

    # If we don't have a target model (inference), the live model MUST pick the action
    acting_preds = target_preds if target_preds is not None else model(obs)
    actions = extractor_fn(acting_preds)

    return target_preds, actions


def double_selector(
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    obs: torch.Tensor,
    extractor_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Double DQN: Live model picks, Target model evaluates.

    Args:
        model (nn.Module): The model to use for action selection.
        target_model (nn.Module): The target model to use for action selection.
        obs (torch.Tensor): The observations to use for action selection.
        extractor_fn (Callable): The function to use for action selection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the target predictions and the actions selected by the model.
    """
    # Skip target compute during actor rollout
    target_preds = target_model(obs) if target_model is not None else None

    live_preds = model(obs)
    actions = extractor_fn(live_preds)

    return target_preds, actions


def with_epsilon_greedy(action_selection_fn: Callable) -> Callable:
    """
    Higher-order function that augments a selector with epsilon-greedy logic.

    Args:
        action_selection_fn (Callable): The function to use for action selection.

    Returns:
        Callable: The action selection function with epsilon-greedy logic.
    """

    def wrapped_selector(
        model: torch.nn.Module,
        target_model: torch.nn.Module,
        obs: torch.Tensor,
        epsilon: float,
        num_actions: int,
        generator: torch.Generator = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Generator]:
        q_values, greedy_actions = action_selection_fn(model, target_model, obs)
        batch_size = obs.shape[0]

        random_actions = torch.randint(
            0, num_actions, (batch_size, 1), generator=generator, device=obs.device
        )
        random_mask = (
            torch.rand((batch_size, 1), generator=generator, device=obs.device)
            < epsilon
        )

        final_actions = torch.where(random_mask, random_actions, greedy_actions)
        return q_values, final_actions, generator

    return wrapped_selector


def get_linear_epsilon(
    step: int, start_eps: float, end_eps: float, decay_steps: int
) -> float:
    """
    Linearly decays epsilon from start_eps to end_eps over decay_steps.

    Args:
        step (int): The current step.
        start_eps (float): The starting epsilon.
        end_eps (float): The ending epsilon.
        decay_steps (int): The number of steps over which to decay epsilon.
    """
    # Calculate the fraction of the way through the decay period (capped at 1.0)
    fraction = min(1.0, float(step) / decay_steps)
    return start_eps - fraction * (start_eps - end_eps)


def get_exponential_epsilon(
    step: int, start_eps: float, end_eps: float, decay_rate: float
) -> float:
    """
    Exponentially decays epsilon, decay rate controls how fast it drops.

    Args:
        step (int): The current step.
        start_eps (float): The starting epsilon.
        end_eps (float): The ending epsilon.
        decay_rate (float): The decay rate.
    """
    return end_eps + (start_eps - end_eps) * math.exp(-1.0 * step / decay_rate)


def get_ape_x_epsilon(
    actor_id: int, num_actors: int, base_eps: float = 0.4, alpha: float = 7.0
) -> float:
    """
    Calculates the fixed epsilon for a specific actor in APE-X.

    Args:
        actor_id (int): The ID of the actor.
        num_actors (int): The total number of actors.
        base_eps (float): The base epsilon value.
        alpha (float): The alpha parameter for the distribution.
    """
    if num_actors <= 1:
        return base_eps
    return base_eps ** (1 + (actor_id / (num_actors - 1)) * alpha)
