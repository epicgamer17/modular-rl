# --- Reusing Gumbel Helper Logic ---
import torch


def get_completed_q_improved_policy(config, node, min_max_stats):
    completedQ = get_completed_q(node, min_max_stats)
    sigma = calculate_gumbel_sigma(
        config.gumbel_cvisit, config.gumbel_cscale, node, completedQ
    )
    eps = 1e-12
    # Ensure network policy is on the same device as sigma/completedQ or CPU
    # Usually completedQ is on device?
    # Let's move to cpu for safety if needed or keep consistency.
    logits = torch.log(node.child_priors + eps)
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
