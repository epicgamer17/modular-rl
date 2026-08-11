# TODO: soft min max stats? efficient_zero.pdf
# TODO: change interface to be more like mctx? where this takes in a tree? or should we just add a visit_count?
def qtrasform_by_min_max(
    q_values: torch.Tensor, min_q: torch.Tensor, max_q: torch.Tensor
) -> torch.Tensor:
    """
    Normalizes Q-values to [0, 1] using the min/max observed in the tree.

    Args:
        q_values: Q-values to normalize [..., Num_Actions].
        min_q: Minimum observed Q-value per batch element [B].
        max_q: Maximum observed Q-value per batch element [B].
    """
    # Reshape min/max for broadcasting if q_values is [B, A]
    # TODO: can we do this somehow without the if statement to minimize branching.
    if q_values.ndim > min_q.ndim:
        min_q = min_q.view(-1, 1)
        max_q = max_q.view(-1, 1)

    span = max_q - min_q
    # Protect against division by zero and handle uninitialized min/max
    # TODO: is this justified by the original paper or is this soft min max stats from efficient zero bleeding through?
    span = torch.where(span > 1e-6, span, torch.ones_like(span))
    return (q_values - min_q) / span


def qtrasform_by_parent_and_siblings():
    pass


def qtransform_completed_by_mix_value():
    pass


def _rescale_qvalues():
    pass


def _complete_qvalues():
    pass


def _compute_mixed_value():
    pass
