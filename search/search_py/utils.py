# --- Reusing Gumbel Helper Logic ---
import torch


def _safe_log_probs(probs: torch.Tensor) -> torch.Tensor:
    """Converts probabilities to logits while keeping exact zeros as -inf."""
    return torch.where(probs > 0, probs.log(), torch.full_like(probs, -float("inf")))


def get_completed_q_improved_policy(
    gumbel_cvisit: int, gumbel_cscale: float, node, min_max_stats
):
    """Compute the improved policy π₀ using clean network logits (Paper Eq. 11).

    The policy improvement target must be computed from the pure network policy
    (π), not the Gumbel-injected child_priors. Using noisy priors as the logit
    baseline would bias the target policy toward the injected noise.

    Args:
        gumbel_cvisit: Gumbel c_visit parameter.
        gumbel_cscale: Gumbel c_scale parameter.
        node: The root DecisionNode after search.
        min_max_stats: Running min/max statistics for Q-value normalization.

    Returns:
        pi0 (Tensor): Softmax-normalised improved policy over all actions.
    """
    completedQ = get_completed_q(node, min_max_stats)
    sigma = calculate_gumbel_sigma(gumbel_cvisit, gumbel_cscale, node, completedQ)
    # Use clean network logits as the base — fall back to child_priors only if
    # network_policy is unavailable (e.g., for internal nodes).
    base_priors = (
        node.network_policy if node.network_policy is not None else node.child_priors
    )
    logits = torch.where(
        base_priors > 0,
        torch.log(base_priors),
        torch.full_like(base_priors, -float("inf")),
    )
    pi0_logits = logits + sigma
    pi0 = torch.softmax(pi0_logits, dim=0)
    return pi0



def get_completed_q(node, min_max_stats):
    v_mix = node.get_v_mix()
    # Initialize with v_mix
    completedQ = torch.full((len(node.child_priors),), min_max_stats.normalize(v_mix))

    # Identify visited actions
    visited_mask = node.child_visits > 0

    if visited_mask.any():
        # Get Q-values for visited actions
        q_vals = node.child_values[visited_mask]

        # Verify min_max_stats.normalize works on tensors
        # Assuming it does (MinMaxStats usually handles tensor ranges if implemented correctly)
        # If not, we might need to map.
        # But MinMaxStats normalization is (val - min) / (max - min).
        # Tensors support this natively.
        normalized_q = min_max_stats.normalize(q_vals)

        completedQ[visited_mask] = normalized_q

    # Handle unvisited nodes with bootstrap value
    if not visited_mask.all():
        bootstrap_val = node.get_child_q_for_unvisited()
        # Normalize bootstrap value too!
        # min_max_stats might not have seen this value if unvisited?
        # But min_max_stats tracks visited values.
        # Standard practice: normalize bootstrap with current stats.
        norm_bootstrap = min_max_stats.normalize(bootstrap_val)
        completedQ[~visited_mask] = norm_bootstrap

    return completedQ


def calculate_gumbel_sigma(gumbel_cvisit, gumbel_cscale, node, completedQ):
    max_visits = node.child_visits.max().item() if len(node.child_visits) > 0 else 0
    return (gumbel_cvisit + max_visits) * gumbel_cscale * completedQ
