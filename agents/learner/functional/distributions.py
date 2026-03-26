import torch
from torch import Tensor


def project_scalars_to_discrete_support(
    scalars: Tensor,
    vmin: float,
    vmax: float,
    bins: int,
) -> Tensor:
    """
    Projects scalar values onto a discrete support grid (two-hot encoding).
    Useful for MuZero-style value/reward targets.
    
    Args:
        scalars: [B, T] or [B] scalar values
        vmin: grid min
        vmax: grid max
        bins: number of atoms in grid
        
    Returns:
        [..., bins] target distribution
    """
    device = scalars.device
    dtype = scalars.dtype
    orig_shape = scalars.shape
    delta_z = (vmax - vmin) / (bins - 1)

    # 1. Flatten into [N]
    x = scalars.reshape(-1).clamp(vmin, vmax)

    # 2. Process
    b = (x - vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    p_u = b - l.float()
    p_l = 1.0 - p_u

    flat_l = l.view(-1, 1).clamp(0, bins - 1)
    flat_u = u.view(-1, 1).clamp(0, bins - 1)
    flat_p_l = p_l.view(-1, 1)
    flat_p_u = p_u.view(-1, 1)

    num_elements = x.shape[0]
    projected = torch.zeros((num_elements, bins), device=device, dtype=dtype)

    projected.scatter_add_(1, flat_l, flat_p_l)
    projected.scatter_add_(1, flat_u, flat_p_u)

    # 3. Unflatten back to [..., bins]
    return projected.view(*orig_shape, bins)
