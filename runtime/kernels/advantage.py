import torch
from typing import Tuple
from core.batch import TransitionBatch

def gae_advantage(
    batch: TransitionBatch,
    next_value: torch.Tensor,
    next_terminated: torch.Tensor,
    gamma: float,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation kernel."""
    rewards = batch.reward
    values = batch.value
    dones = batch.done if batch.done is not None else batch.terminated
    
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - next_terminated.float()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )

    returns = advantages + values
    return advantages, returns

def td_lambda_advantage(
    batch: TransitionBatch,
    next_value: torch.Tensor,
    next_terminated: torch.Tensor,
    gamma: float,
    td_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TD(lambda) returns and advantages kernel."""
    rewards = batch.reward
    values = batch.value
    dones = batch.done if batch.done is not None else batch.terminated
    
    T, N = rewards.shape
    returns = torch.zeros_like(rewards)
    last_return = next_value * (1.0 - next_terminated.float())
    
    for t in reversed(range(T)):
        if t == T - 1:
            mask = 1.0 - next_terminated.float()
            next_v = next_value
        else:
            mask = 1.0 - dones[t].float()
            next_v = values[t + 1]
            
        returns[t] = rewards[t] + gamma * mask * (
            (1 - td_lambda) * next_v + td_lambda * last_return
        )
        last_return = returns[t]
        
    advantages = returns - values
    return advantages, returns

def mc_advantage(
    batch: TransitionBatch,
    next_value: torch.Tensor,
    next_terminated: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full Monte Carlo returns kernel."""
    rewards = batch.reward
    values = batch.value
    dones = batch.done if batch.done is not None else batch.terminated
    
    T, N = rewards.shape
    returns = torch.zeros_like(rewards)
    last_return = next_value * (1.0 - next_terminated.float())
    
    for t in reversed(range(T)):
        returns[t] = rewards[t] + gamma * last_return
        last_return = returns[t]
        if t > 0:
            last_return = last_return * (1.0 - dones[t-1].float())
            
    advantages = returns - values
    return advantages, returns
