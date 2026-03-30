import itertools
import math
import bisect
import torch
import torch.nn.init as init
from torch import nn, Tensor, optim
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Any, Callable


@torch.inference_mode()
def get_flat_dim(module: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Foolproof calculation of flattened output dimension using a dummy pass."""
    dummy_input = torch.zeros(1, *input_shape)
    dummy_output = module(dummy_input)
    return dummy_output.flatten(1, -1).shape[1]


if TYPE_CHECKING:
    from configs.base import Config

from utils.schedule import ScheduleConfig, create_schedule, Schedule


def create_optimizer(params, config, sub_config_parent=None):
    parent = sub_config_parent if sub_config_parent is not None else config

    # Safely get optimizer class and LR, falling back to main config
    opt_cls = getattr(parent, "optimizer", getattr(config, "optimizer", Adam))
    lr = getattr(parent, "learning_rate", getattr(config, "learning_rate", 1e-3))

    if opt_cls == Adam:
        return Adam(
            params=params,
            lr=lr,
            # Fall back to root config if parent lacks the attribute
            eps=getattr(parent, "adam_epsilon", getattr(config, "adam_epsilon", 1e-8)),
            weight_decay=getattr(
                parent, "weight_decay", getattr(config, "weight_decay", 0.0)
            ),
        )
    elif opt_cls == SGD:
        return SGD(
            params=params,
            lr=lr,
            momentum=getattr(parent, "momentum", getattr(config, "momentum", 0.0)),
            weight_decay=getattr(
                parent, "weight_decay", getattr(config, "weight_decay", 0.0)
            ),
        )
    else:
        raise ValueError(f"Unsupported optimizer class: {opt_cls}")


def support_to_scalar(
    probabilities: torch.Tensor, support_size: int, eps: float = 0.001
):
    """
    Convert categorical probabilities over the support [-support_size .. +support_size]
    into scalar(s) using the inverse of the MuZero transformation.

    Args:
        probabilities: Tensor of shape (L,) or (B, L) where L == 2*support_size + 1.
        support_size: integer support size.
        eps: small epsilon used by MuZero (default 0.001).

    Returns:
        Tensor of shape () (scalar) if input was 1D, or (B,) for batched input.
    """
    if probabilities.dim() == 1:
        probs = probabilities.unsqueeze(0)  # shape (1, L)
        squeeze_out = True
    else:
        probs = probabilities
        squeeze_out = False

    batch, L = probs.shape
    assert L == 2 * support_size + 1, "probabilities length must equal 2*support_size+1"

    device = probs.device
    dtype = probs.dtype

    support = torch.arange(
        -support_size, support_size + 1, device=device, dtype=dtype
    ).unsqueeze(
        0
    )  # (1, L)
    z = torch.sum(
        probs * support, dim=1
    )  # expected value on transformed scale, shape (B,)

    # inverse transform from MuZero appendix:
    # f^{-1}(z) = sign(z) * ( ((sqrt(1 + 4*eps*(|z| + 1 + eps)) - 1) / (2*eps))^2 - 1 )
    sign = torch.sign(z)
    abs_z = torch.abs(z)
    inner = 1.0 + 4.0 * eps * (abs_z + 1.0 + eps)
    inv = sign * (((torch.sqrt(inner) - 1.0) / (2.0 * eps)) ** 2 - 1.0)

    if squeeze_out:
        return inv.squeeze(0)
    return inv  # shape (B,)


def scalar_to_support(x: torch.Tensor | float, support_size: int, eps: float = 0.001):
    """
    Convert scalar(s) into a categorical (2*support_size+1)-vector on the MuZero transformed scale.
    Handles tensors of arbitrary shape, returning (..., 2*support_size+1).
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    original_shape = x.shape
    x_flat = x.reshape(-1)

    device = x_flat.device
    num_elements = x_flat.shape[0]
    L = 2 * support_size + 1

    # forward transform from MuZero appendix:
    # f(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
    sign = torch.sign(x_flat)
    abs_x = torch.abs(x_flat)
    x_trans = sign * (torch.sqrt(abs_x + 1.0) - 1.0) + eps * x_flat

    # clamp into support range
    x_trans = torch.clamp(x_trans, -support_size, support_size)

    floor = torch.floor(x_trans)
    prob = x_trans - floor  # fractional part, in [0,1)

    out = torch.zeros((num_elements, L), device=device, dtype=torch.float32)

    idx_lower = (floor + support_size).long()  # index for floor
    idx_upper = idx_lower + 1  # index for floor+1

    flat_idx = torch.arange(num_elements, device=device)

    # assign (1 - frac) to floor bin
    out[flat_idx, idx_lower] = 1.0 - prob

    # assign frac to upper bin if within range
    valid_upper_mask = idx_upper <= (L - 1)
    if valid_upper_mask.any():
        out[flat_idx[valid_upper_mask], idx_upper[valid_upper_mask]] = prob[
            valid_upper_mask
        ]

    # Reshape back to original shape + support dimension
    return out.view(*original_shape, L)


def scale_gradient(tensor, scale):
    """
    Scales the gradient for the backward pass without changing the forward pass.
    Args:
        tensor (torch.Tensor): The input tensor.
        scale (float): The scaling factor for the gradient.
    """
    return tensor * scale + tensor.detach() * (1 - scale)


_epsilon = 1e-7

from typing import Any, Callable, Optional, Tuple

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def calculate_padding(i: int, k: int, s: int) -> Tuple[int, int]:
    """Calculate both padding sizes along 1 dimension for a given input length, kernel length, and stride

    Args:
        i (int): input length
        k (int): kernel length
        s (int): stride

    Returns:
        (p_1, p_2): where p_1 = p_2 - 1 for uneven padding and p_1 == p_2 for even padding
    """

    o = math.ceil(i / s)
    p = max(0, (o - 1) * s + k - i)
    p_1 = p // 2
    p_2 = (p + 1) // 2
    return (p_1, p_2)


def calculate_same_padding(i, k, s) -> Tuple[None | Tuple[int], None | str | Tuple]:
    """Calculate pytorch inputs for same padding
    Args:
        i (int, int) or int: (h, w) or (w, w)
        k (int, int) or int: (k_h, k_w) or (k, k)
        s (int, int) or int: (s_h, s_w) or (s, s)
    Returns:
        Tuple[manual_pad_padding, torch_conv2d_padding_input]: Either the manual padding that must be applied (first element of tuple) or the input to the torch padding argument of the Conv2d layer
    """

    if s == 1:
        return None, "same"
    h, w = unpack(i)
    k_h, k_w = unpack(k)
    s_h, s_w = unpack(s)
    p_h = calculate_padding(h, k_h, s_h)
    p_w = calculate_padding(w, k_w, s_w)
    if p_h[0] == p_h[1] and p_w[0] == p_w[1]:
        return None, (p_h[0], p_w[0])
    else:
        # not torch compatiable, manually pad with torch.nn.functional.pad
        return (*p_w, *p_h), None


def generate_layer_widths(widths: list[int], max_num_layers: int) -> list[Tuple[int]]:
    """Create all possible combinations of widths for a given number of layers"""
    width_combinations = []

    for i in range(0, max_num_layers):
        width_combinations.extend(itertools.combinations_with_replacement(widths, i))

    return width_combinations


def prepare_activations(activation: str | nn.Module | Callable):
    """
    Returns an activation module from a string name or returns the input if already a module/callable.
    """
    if isinstance(activation, nn.Module) or (
        callable(activation) and not isinstance(activation, str)
    ):
        return activation

    # print("Activation to prase: ", activation)
    if activation == "linear":
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "relu6":
        return nn.ReLU6()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "soft_sign":
        return nn.Softsign()
    elif activation == "silu" or activation == "swish":
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    # elif activation == "log_sigmoid":
    #     return nn.LogSigmoid()
    elif activation == "hard_sigmoid":
        return nn.Hardsigmoid()
    # elif activation == "hard_silu" or activation == "hard_swish":
    #     return nn.Hardswish()
    # elif activation == "hard_tanh":
    #     return nn.Hardtanh()
    elif activation == "elu":
        return nn.ELU()
    # elif activation == "celu":
    #     return nn.CELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "gelu":
        return nn.GELU()
    # elif activation == "glu":
    #     return nn.GLU()

    raise ValueError(f"Activation {activation} not recognized")


def calc_units(shape):
    shape = tuple(shape)
    if len(shape) == 1:
        return shape + shape
    if len(shape) == 2:
        # dense layer -> (in_channels, out_channels)
        return shape
    else:
        # conv_layer (Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (input_depth, depth, ...)
        in_units = shape[1]
        out_units = shape[0]
        c = 1
        for dim in shape[2:]:
            c *= dim
        return (c * in_units, c * out_units)


# modules/network_utils.py (New File)
from torch import nn
from typing import Literal


def build_normalization_layer(
    norm_type: Literal["batch", "layer", "none"], num_features: int, dim: int
) -> nn.Module:
    """
    Builds the specified normalization layer.

    Args:
        norm_type: The type of normalization ('batch', 'layer', 'none').
        num_features: The number of features (channels for conv, width for dense).
        dim: The dimension of the input tensor (2 for conv/2D, 1 for dense/1D).
    """
    if norm_type == "batch":
        if dim == 2:
            return nn.BatchNorm2d(num_features)
        elif dim == 1:
            # Batch norm for 1D (Dense) layers
            return nn.BatchNorm1d(num_features)
        else:
            raise ValueError(f"Batch norm for {dim}D not supported.")
    elif norm_type == "layer":
        # nn.LayerNorm expects a list of shape for LayerNorm on last dim(s).
        # We assume the layer is applied across the feature dimension.
        return nn.LayerNorm(num_features)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


# Existing unpack function from your original code (in utils)
def unpack(x: int | Tuple):
    # ... (same as your original implementation)
    if isinstance(x, Tuple):
        assert len(x) == 2
        return x
    else:
        try:
            x = int(x)
            return x, x
        except Exception as e:
            print(f"error converting {x} to int: ", e)


def _normalize_hidden_state(S: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Normalizes the hidden state tensor as described in the MuZero paper (Appendix A.2).
    Applies Min-Max scaling to [0, 1] globally across all feature dimensions per sample.
    
    Args:
        S: Hidden state tensor of shape (B, *) or (B, C, H, W).
        eps: Small epsilon to prevent division by zero.
    """
    # Flatten all dimensions except batch: [B, W] or [B, C, H, W] -> [B, -1]
    batch_size = S.shape[0]
    S_flat = S.view(batch_size, -1)
    
    # Calculate min/max across the flattened feature dimension
    min_val = S_flat.min(dim=1, keepdim=True)[0]
    max_val = S_flat.max(dim=1, keepdim=True)[0]
    
    # Expand back to original shape for broadcasting
    # (min_val and max_val are [B, 1], broadcasting to (B, *) is automatic except for 4D)
    if S.dim() == 4:
        min_val = min_val.view(batch_size, 1, 1, 1)
        max_val = max_val.view(batch_size, 1, 1, 1)
        
    denominator = max_val - min_val
    denominator = torch.where(denominator < eps, torch.ones_like(denominator), denominator)
    
    return (S - min_val) / denominator


from dataclasses import dataclass
import torch
import torch.nn.functional as F


def get_optimal_device() -> torch.device:
    """Returns the optimal available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# Helper function to use across your framework
def get_clean_state_dict(model: torch.nn.Module) -> dict:
    """Strips the '_orig_mod.' prefix added by torch.compile."""
    state_dict = model.state_dict()
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


@torch.inference_mode()
def update_target_network(
    online: torch.nn.Module | dict[str, torch.Tensor],
    target_net: torch.nn.Module,
    tau: float = 1.0,
) -> None:
    """
    Standardized, high-performance target network synchronization.
    Uses in-place .data.copy_() to avoid memory fragmentation and re-allocation.
    
    Args:
        online: Source module OR state_dict.
        target_net: Destination network.
        tau: Sync coefficient. 1.0 = Hard Sync, <1.0 = Soft Sync (EMA).
    """
    if isinstance(online, torch.nn.Module):
        online_state = get_clean_state_dict(online)
    else:
        online_state = online

    target_state = target_net.state_dict()

    for k, online_val in online_state.items():
        if k in target_state:
            target_val = target_state[k]
            if tau == 1.0 or not target_val.is_floating_point():
                # Hard Sync or non-float buffer (e.g. step count)
                target_val.data.copy_(online_val.data)
            else:
                # Soft Sync (EMA): target = tau * online + (1-tau) * target
                target_val.data.copy_(
                    tau * online_val.data + (1.0 - tau) * target_val.data
                )


def get_uncompiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Safely extracts the uncompiled base model to allow pickling across processes.
    Handles both module-level and method-level compilation without mutating the original.
    """
    import copy

    # If the model was compiled at the module level (torch.compile(model))
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    # If the model used method-level compilation (e.g. self.obs_inference = torch.compile(...))
    # This prevents PicklingError when passing the model to worker processes like Tester.
    has_compiled_methods = any(
        hasattr(model, m) and m in model.__dict__
        for m in [
            "obs_inference",
            "hidden_state_inference",
            "afterstate_inference",
            "learner_inference",
            "step",
        ]
    )

    if has_compiled_methods:
        # Create a shallow copy of the model itself
        m_copy = copy.copy(model)
        # Create a shallow copy of its __dict__ so we don't mutate the original instance's methods
        m_copy.__dict__ = copy.copy(model.__dict__)

        # Remove the compiled method instances from __dict__ so pickle falls back to class methods
        compiled_methods = [
            "obs_inference",
            "hidden_state_inference",
            "afterstate_inference",
            "learner_inference",
            "step",
        ]
        for attr in compiled_methods:
            if attr in m_copy.__dict__:
                del m_copy.__dict__[attr]

        return m_copy

    return model


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: Any
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Returns a learning rate scheduler based on the configuration.
    """
    # Unify on 'lr_schedule' as requested by the user
    schedule = getattr(config, "lr_schedule", None)

    # Determine scheduler type (string or from ScheduleConfig object)
    if schedule is not None and hasattr(schedule, "type"):
        scheduler_type = schedule.type
    else:
        scheduler_type = schedule

    if scheduler_type is None or scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

    if scheduler_type == "linear":
        # Total iterations usually comes from training_steps
        total_iters = getattr(config, "training_steps", 1000)

        # Support decay to a final factor (default to 0.1 for backward compatibility)
        # If using ScheduleConfig, derive the factor from initial/final values
        end_factor = 0.1
        if schedule is not None and hasattr(schedule, "final") and hasattr(schedule, "initial"):
            if schedule.initial is not None and schedule.initial != 0 and schedule.final is not None:
                end_factor = schedule.final / schedule.initial
        else:
            # Fallback to direct config attribute if available
            end_factor = getattr(config, "lr_final_factor", 0.1)

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_iters
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=getattr(config, "lr_step_size", 100), gamma=0.1
        )

    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
