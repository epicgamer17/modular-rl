import torch
import torch.nn.functional as F

from .value_target import compute_v_td_target


# TODO: I think truncation right now works for single env trajectories without auto resetting, but would fail for vectorized envs with auto resetting. as the next_q_values would be on the resetted state instead of the one from the info. verify this, and if it is an issue, unify the vectorization logic in our buffer to work with offline buffers as well somehow.


def compute_q_td_target(
    next_q_values: torch.Tensor,  # [B, A]
    next_actions: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    terminated: torch.Tensor,  # [B]
    gamma: torch.Tensor,  # [B]
) -> torch.Tensor:
    """
    Calculates the TD target for scalar Q-values.
    Composes the V-target function by extracting the value of the next state.

    Args:
        next_q_values: Q-values of the next states.
        next_actions: Indices of the actions taken in the next states (greedy for Q-learning, sampled for SARSA).
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.

    Returns:
        torch.Tensor: The detached TD target of shape [B]. Gradients will NOT flow through this tensor.
    """
    assert (
        next_q_values.ndim == 2
    ), f"Expected 2D next_q_values [B, A], got {next_q_values.shape}"
    assert (
        next_actions.ndim == 1
    ), f"Expected [B] next_actions, got {next_actions.shape}"

    # 1. Extract the Q-value of the selected next action -> This IS V(s')
    next_values = torch.gather(next_q_values, 1, next_actions.unsqueeze(-1)).squeeze(-1)

    # 2. Compute standard V-target
    target = compute_v_td_target(next_values, rewards, terminated, gamma)
    return target.detach()


def compute_categorical_q_td_target(
    next_logits: torch.Tensor,  # [B, A, Atoms]
    next_actions: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    terminated: torch.Tensor,  # [B]
    gamma: torch.Tensor,  # [B]
    support: torch.Tensor,  # [Atoms]
    v_min: float,
    v_max: float,
    atom_size: int,
) -> torch.Tensor:
    """
    Calculates the projected Categorical TD target distribution (C51 style).

    This function handles both 1-step and N-step TD targets. For N-step,
    the `rewards` should be the pre-computed discounted sum of rewards,
    and `gamma` should be the pre-computed effective discount factor (gamma^n).

    Args:
        next_logits: Logits of the next states.
        next_actions: Indices of the actions taken in the next states.
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.
        support: Support values for the distribution.
        v_min: The minimum value of the support.
        v_max: The maximum value of the support.
        atom_size: The number of atoms in the support.

    Returns:
        torch.Tensor: The detached projected Categorical TD target distribution [B, Atoms]. Gradients will NOT flow through this tensor.
    """
    assert (
        next_logits.ndim == 3
    ), f"Expected 3D next_logits [B, A, Atoms], got {next_logits.shape}"
    assert (
        next_actions.ndim == 1
    ), f"Expected [B] next_actions, got {next_actions.shape}"

    # 1. Get probabilities of the next states
    next_probs = F.softmax(next_logits, dim=-1)

    # 2. Gather the probabilities for the chosen next actions
    next_actions_expanded = next_actions.view(-1, 1, 1).expand(-1, -1, atom_size)
    next_probs_a = next_probs.gather(1, next_actions_expanded).squeeze(1)  # [B, Atoms]

    # 3. Compute the target support (Tz) [B, Atoms]
    support_b = support.unsqueeze(0)
    rewards_b = rewards.unsqueeze(1)
    gamma_b = gamma.unsqueeze(1)
    term_b = terminated.unsqueeze(1)

    Tz = rewards_b + gamma_b * support_b * (1 - term_b.float())
    Tz = Tz.clamp(min=v_min, max=v_max)

    # 4. Compute projection bins
    dz = (v_max - v_min) / (atom_size - 1)
    b = (Tz - v_min) / dz
    l = b.floor().long()
    u = b.ceil().long()

    # Handle boundary conditions where the target falls exactly on a bin
    l[(u > 0) & (l == u)] -= 1
    u[(l < (atom_size - 1)) & (l == u)] += 1

    # 5. Distribute probabilities onto the fixed support (Projection)
    batch_size = rewards.size(0)
    m = rewards.new_zeros(batch_size, atom_size)
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

    # Flatten views for categorical projection
    m_flat = m.view(-1)
    offset_l = (l + offset).view(-1)
    offset_u = (u + offset).view(-1)

    prob_lower = (next_probs_a * (u.float() - b)).view(-1)
    prob_upper = (next_probs_a * (b - l.float())).view(-1)

    # Index Add becomes clean and fast:
    m_flat.index_add_(0, offset_l, prob_lower)
    m_flat.index_add_(0, offset_u, prob_upper)

    return m.detach()
