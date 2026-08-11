from tensordict import TensorDict
import torch


# TODO: soft min max stats? efficient_zero.pdf
# TODO: change interface to be more like mctx? where this takes in a tree? or should we just add a visit_count?
def qtransform_by_min_max(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    min_value: torch.Tensor,  # [B] or scalar
    max_value: torch.Tensor,  # [B] or scalar
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Returns Q-values normalized by the given `min_value` and `max_value`.

    Unvisited actions are assigned zero Q-value score after normalization.
    Shape: [B, num_actions].
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)

    # Extract Q-values and visit counts for candidate actions: [B, A]
    qvalues = tree["children_values"][batch_range, parent_nodes]
    visit_counts = tree["children_visits"][batch_range, parent_nodes]

    # TODO: can we do this without the if statements
    min_val = min_value.unsqueeze(-1) if min_value.ndim == 1 else min_value
    max_val = max_value.unsqueeze(-1) if max_value.ndim == 1 else max_value

    # Assign min_value to unvisited actions before scaling
    value_score = torch.where(visit_counts > 0, qvalues, min_val)

    # Safe normalization
    denom = torch.clamp(max_val - min_val, min=epsilon)
    value_score = (value_score - min_val) / denom

    # Unvisited actions explicitly get 0.0 Q-value
    return torch.where(visit_counts > 0, value_score, 0.0)


def qtransform_by_parent_and_siblings(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Returns Q-values normalized by min/max over V(parent) and visited Q-values.

    Unvisited actions will have zero Q-value.
    Shape: [B, num_actions].
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)

    qvalues = tree["children_values"][batch_range, parent_nodes]  # [B, A]
    visit_counts = tree["children_visits"][batch_range, parent_nodes]  # [B, A]
    node_value = tree["node_values"][batch_range, parent_nodes].unsqueeze(-1)  # [B, 1]

    # Safe Q-values: unvisited actions take parent node_value
    safe_qvalues = torch.where(visit_counts > 0, qvalues, node_value)

    # Dynamic min and max across siblings and parent
    min_value = torch.minimum(node_value, safe_qvalues.amin(dim=-1, keepdim=True))
    max_value = torch.maximum(node_value, safe_qvalues.amax(dim=-1, keepdim=True))

    completed_by_min = torch.where(visit_counts > 0, qvalues, min_value)
    denom = torch.clamp(max_value - min_value, min=epsilon)
    normalized = (completed_by_min - min_value) / denom

    # Unvisited actions get 0.0
    return torch.where(visit_counts > 0, normalized, 0.0)


def qtransform_completed_by_mix_value(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Computes completed Q-values for Gumbel MuZero.

    Missing Q-values for unvisited actions are replaced by the mixed value (V_mix).
    Applies linear scaling: (maxvisit_init + max(visit_counts)) * value_scale * completed_qvalues.
    Shape: [B, num_actions].
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)

    qvalues = tree["children_values"][batch_range, parent_nodes]  # [B, A]
    visit_counts = tree["children_visits"][batch_range, parent_nodes]  # [B, A]
    raw_value = tree["raw_values"][batch_range, parent_nodes]  # [B]
    prior_logits = tree["children_prior_logits"][batch_range, parent_nodes]  # [B, A]
    prior_probs = F.softmax(prior_logits, dim=-1)  # [B, A]

    if use_mixed_value:
        value = _compute_mixed_value(
            raw_value,
            qvalues=qvalues,
            visit_counts=visit_counts,
            prior_probs=prior_probs,
        )  # [B]
    else:
        value = raw_value

    completed_qvalues = _complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value
    )  # [B, A]

    if rescale_values:
        completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)

    maxvisit = visit_counts.amax(dim=-1, keepdim=True).to(completed_qvalues.dtype)
    visit_scale = maxvisit_init + maxvisit
    return visit_scale * value_scale * completed_qvalues


def _rescale_qvalues(qvalues: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Rescales completed Q-values to [0, 1] per batch item."""
    min_value = qvalues.amin(dim=-1, keepdim=True)
    max_value = qvalues.amax(dim=-1, keepdim=True)
    denom = torch.clamp(max_value - min_value, min=epsilon)
    return (qvalues - min_value) / denom


def _complete_qvalues(
    qvalues: torch.Tensor,
    *,
    visit_counts: torch.Tensor,
    value: torch.Tensor,  # [B]
) -> torch.Tensor:
    """Replaces unvisited actions with value [B, 1]."""
    val_expanded = value.unsqueeze(-1) if value.ndim == 1 else value
    return torch.where(visit_counts > 0, qvalues, val_expanded)


def _compute_mixed_value(
    raw_value: torch.Tensor,  # [B]
    qvalues: torch.Tensor,  # [B, A]
    visit_counts: torch.Tensor,  # [B, A]
    prior_probs: torch.Tensor,  # [B, A]
) -> torch.Tensor:
    """Interpolates raw_value and prior-weighted Q-values as defined in Gumbel MuZero."""
    sum_visit_counts = visit_counts.sum(dim=-1).to(qvalues.dtype)  # [B]

    # Ensure non-zero prior_probs to avoid Division-by-Zero / NaNs
    tiny = torch.finfo(prior_probs.dtype).tiny
    safe_priors = torch.clamp(prior_probs, min=tiny)

    # Sum probabilities of visited actions
    visited_mask = visit_counts > 0
    sum_probs = torch.where(visited_mask, safe_priors, 0.0).sum(dim=-1, keepdim=True)
    sum_probs_safe = torch.where(sum_probs > 0, sum_probs, 1.0)

    # Re-normalize priors over visited actions only
    normalized_priors = safe_priors / sum_probs_safe
    weighted_q = torch.where(visited_mask, normalized_priors * qvalues, 0.0).sum(
        dim=-1
    )  # [B]

    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1.0)
