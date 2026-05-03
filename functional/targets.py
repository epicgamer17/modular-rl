import torch
import torch.nn.functional as F


def standard_td_target(
    next_q_values: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: torch.Tensor,
):
    """
    Calculates the standard 1-step TD target.
    Args:
        next_q_values (torch.Tensor): Tensor of shape (batch_size, num_actions) containing the Q-values of the next states.
        next_actions (torch.Tensor): Tensor of shape (batch_size, 1) containing the indices of the actions taken in the next states.
        rewards (torch.Tensor): Tensor of shape (batch_size, 1) containing the rewards.
        terminated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes terminated.
        gamma (torch.Tensor): Discount factor.
    Returns:
        torch.Tensor: The standard 1-step TD target.
    """
    max_q_next = torch.gather(next_q_values, 1, next_actions)
    gamma = gamma.view(-1, 1)
    return rewards + gamma * max_q_next * (1 - terminated.float())


def n_step_td_target(
    next_q_values: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: torch.Tensor,
):
    """
    Calculates the N-step TD target using an effective gamma.
    Args:
        next_q_values (torch.Tensor): Tensor of shape (batch_size, num_actions) containing the Q-values of the next states.
        next_actions (torch.Tensor): Tensor of shape (batch_size, 1) containing the indices of the actions taken in the next states.
        rewards (torch.Tensor): Tensor of shape (batch_size, 1) containing the rewards.
        terminated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes terminated.
        gamma (torch.Tensor): Discount factor.
        n_steps (int): The number of steps to use for the TD target.
    Returns:
        torch.Tensor: The N-step TD target.
    """
    max_q_next = torch.gather(next_q_values, 1, next_actions)
    effective_gamma = gamma.view(-1, 1)
    return rewards + effective_gamma * max_q_next * (1 - terminated.float())


def categorical_td_target(
    next_logits: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
    atom_size: int,
):
    """
    Calculates the projected Categorical TD target distribution.
    Args:
        next_logits (torch.Tensor): Tensor of shape (batch_size, num_actions, atom_size) containing the logits of the next states.
        next_actions (torch.Tensor): Tensor of shape (batch_size, 1) containing the indices of the actions taken in the next states.
        rewards (torch.Tensor): Tensor of shape (batch_size, 1) containing the rewards.
        terminated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes terminated.
        gamma (torch.Tensor): Tensor of shape (batch_size, 1) containing the discount factors.
        support (torch.Tensor): Tensor of shape (atom_size,) containing the support values.
        v_min (float): The minimum value of the support.
        v_max (float): The maximum value of the support.
        atom_size (int): The number of atoms in the support.
    Returns:
        torch.Tensor: The projected Categorical TD target distribution.
    """
    batch_size = rewards.size(0)

    # 1. Get probabilities of the next states
    next_probs = F.softmax(next_logits, dim=-1)

    # 2. Gather the probabilities for the chosen next actions
    # next_actions is [B, 1], expand to [B, 1, Atoms] to match next_probs
    next_actions_expanded = next_actions.unsqueeze(-1).expand(-1, -1, atom_size)
    next_probs_a = next_probs.gather(1, next_actions_expanded).squeeze(1)  # [B, Atoms]

    # 3. Compute the target support (Tz)
    rewards = rewards.view(-1, 1)
    terminated = terminated.view(-1, 1)
    gamma = gamma.view(-1, 1)
    Tz = rewards + gamma * support.view(1, -1) * (1 - terminated.float())
    Tz = Tz.clamp(min=v_min, max=v_max)

    # 4. Compute projection bins
    dz = (v_max - v_min) / (atom_size - 1)
    b = (Tz - v_min) / dz
    l = b.floor().long()
    u = b.ceil().long()

    # Handle boundary conditions where the target falls exactly on a bin
    l[(u > 0) & (l == u)] -= 1
    u[(l < (atom_size - 1)) & (l == u)] += 1

    # 5. Distribute probabilities onto the fixed support
    m = torch.zeros(batch_size, atom_size, device=rewards.device)
    offset = (
        torch.linspace(
            0,
            (batch_size - 1) * atom_size,
            batch_size,
            dtype=torch.long,
            device=rewards.device,
        )
        .unsqueeze(1)
        .expand(batch_size, atom_size)
    )

    m.view(-1).index_add_(
        0, (l + offset).view(-1), (next_probs_a * (u.float() - b)).view(-1)
    )
    m.view(-1).index_add_(
        0, (u + offset).view(-1), (next_probs_a * (b - l.float())).view(-1)
    )

    return m  # This is the target probability distribution
