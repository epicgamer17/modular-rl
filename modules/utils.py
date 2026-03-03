import itertools
import math
import bisect
import torch
import torch.nn.init as init
from torch import nn, Tensor, optim
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.base import Config

from utils.schedule import ScheduleConfig, Schedule, create_schedule


class LinearLR:
    def __init__(self, training_steps: int):
        self.training_steps = training_steps

    def __call__(self, current_step: int):
        return max(
            0.0,
            float(self.training_steps - current_step)
            / float(max(1, self.training_steps)),
        )


class StepWiseLR:
    def __init__(self, steps: list, values: list, initial_lr: float):
        self.steps = steps
        self.values = values
        self.initial_lr = initial_lr

    def __call__(self, current_step: int):
        idx = bisect.bisect_right(self.steps, current_step)
        if idx == 0:
            return 1.0
        else:
            return self.values[idx - 1] / self.initial_lr


class ConstantLR:
    def __call__(self, _):
        return 1.0


class ScheduleLRScheduler(optim.lr_scheduler.LRScheduler):
    """
    Wraps a Schedule to work with PyTorch's LR scheduler interface.
    """

    def __init__(self, optimizer: optim.Optimizer, schedule: Schedule):
        self.schedule = schedule
        self._initial_lr = schedule.config.initial
        super().__init__(optimizer)

    def get_lr(self):
        if self._initial_lr is None or self._initial_lr == 0:
            return self.base_lrs
        factor = self.schedule.get_value() / self._initial_lr
        return [lr * factor for lr in self.base_lrs]

    def step(self, *args, **kwargs):
        self.schedule.step()
        super().step(*args, **kwargs)


def get_lr_scheduler(optimizer: optim.Optimizer, config: "Config"):
    schedule_config = config.lr_schedule
    if schedule_config is None:
        return optim.lr_scheduler.LambdaLR(optimizer, ConstantLR())

    schedule = create_schedule(schedule_config)
    return ScheduleLRScheduler(optimizer, schedule)


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


# def support_to_scalar(probabilities, support_size):
#     """
#     Transform a categorical representation (1D probs vector)
#     into a scalar.
#     """
#     assert probabilities.dim() == 1, "probabilities must be 1D"
#     assert probabilities.shape[0] == 2 * support_size + 1, "shape mismatch"

#     # support values from -support_size to +support_size
#     support = torch.arange(
#         -support_size,
#         support_size + 1,
#         device=probabilities.device,
#         dtype=torch.float32,
#     )
#     x = torch.sum(support * probabilities)

#     # Invert the scaling (from paper appendix)
#     x = torch.sign(x) * (
#         ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
#         ** 2
#         - 1
#     )
#     return x


# def scalar_to_support(x, support_size):
#     # raise NotImplementedError  # "this is bugged and outputs a uniform instead of a 2-hot"
#     """
#     Transform a scalar (float or 0D tensor)
#     into categorical probs (1D tensor of length 2*support_size+1).
#     """
#     if not torch.is_tensor(x):
#         x = torch.tensor(x, dtype=torch.float32)

#     # Reduce the scale
#     x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

#     # Clamp into support range
#     x = torch.clamp(x, -support_size, support_size)

#     floor = torch.floor(x)
#     prob = x - floor

#     probabilities = torch.zeros(2 * support_size + 1, device=x.device)

#     # distribute mass between floor and floor+1
#     probabilities[int(floor + support_size)] = 1 - prob
#     if floor + 1 <= support_size:
#         probabilities[int(floor + support_size + 1)] = prob

#     # probabilities = torch.softmax(logits, dim=0)
#     return probabilities


# FOR BATCHED INPUTS
# def support_to_scalar(probabilities, support_size):
#     """
#     Transform a categorical representation to a scalar
#     See paper appendix Network Architecture
#     """
#     # Decode to a scalar
#     # print(probabilities.shape)
#     support = (
#         torch.tensor([x for x in range(-support_size, support_size + 1)])
#         .expand(probabilities.shape)
#         .float()
#         .to(device=probabilities.device)
#     )
#     x = torch.sum(support * probabilities, dim=1, keepdim=True)

#     # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
#     x = torch.sign(x) * (
#         ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
#         ** 2
#         - 1
#     )
#     return x


# def scalar_to_support(x, support_size):
#     """
#     Transform a scalar to a categorical representation with (2 * support_size + 1) categories
#     See paper appendix Network Architecture
#     """
#     # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
#     x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

#     # Encode on a vector
#     x = torch.clamp(x, -support_size, support_size)
#     floor = x.floor()
#     prob = x - floor
#     logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
#     logits.scatter_(
#         2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
#     )
#     indexes = floor + support_size + 1
#     prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
#     indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
#     logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
#     probabilities = logits.softmax(dim=2)
#     # print(probabilities.shape)
#     return probabilities


def scale_gradient(tensor, scale):
    """
    Scales the gradient for the backward pass without changing the forward pass.
    Args:
        tensor (torch.Tensor): The input tensor.
        scale (float): The scaling factor for the gradient.
    """
    return tensor * scale + tensor.detach() * (1 - scale)


def zero_weights_initializer(m: nn.Module) -> None:
    """Initializes the weights and biases of a layer to zero."""
    if hasattr(m, "weight") and m.weight is not None:
        init.constant_(m.weight, 0.0)
    if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias, 0.0)


def one_hundredth_initializer(m: nn.Module) -> None:
    """Initializes the weights to uniform(-0.01, 0.01)."""
    if hasattr(m, "weight") and m.weight is not None:
        init.uniform_(m.weight, -0.01, 0.01)
    if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias, 0.0)


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


def prepare_kernel_initializers(kernel_initializer: str, output_layer: bool = False):
    if kernel_initializer == "pytorch_default":
        return None
    if kernel_initializer == "glorot_uniform":
        return nn.init.xavier_uniform_
    elif kernel_initializer == "glorot_normal":
        return nn.init.xavier_normal_
    elif kernel_initializer == "he_uniform":
        # return lambda tensor: nn.init.kaiming_uniform_(tensor, nonlinearity="relu")
        return nn.init.kaiming_uniform_
    elif kernel_initializer == "he_normal":
        # return lambda tensor: nn.init.kaiming_normal_(tensor, nonlinearity="relu")
        return nn.init.kaiming_normal_
    elif kernel_initializer == "variance_baseline":
        return VarianceScaling()
    elif kernel_initializer == "variance_0.1":
        return VarianceScaling(scale=0.1)
    elif kernel_initializer == "variance_0.3":
        return VarianceScaling(scale=0.3)
    elif kernel_initializer == "variance_0.8":
        return VarianceScaling(scale=0.8)
    elif kernel_initializer == "variance_3":
        return VarianceScaling(scale=3)
    elif kernel_initializer == "variance_5":
        return VarianceScaling(scale=5)
    elif kernel_initializer == "variance_10":
        return VarianceScaling(scale=10)
    # TODO
    # elif kernel_initializer == "lecun_uniform":
    #     return LecunUniform(seed=np.random.seed())
    # elif kernel_initializer == "lecun_normal":
    #     return LecunNormal(seed=np.random.seed())
    elif kernel_initializer == "orthogonal":
        return nn.init.orthogonal_
    elif kernel_initializer == "one_hundredth":
        return one_hundredth_initializer

    raise ValueError(f"Invalid kernel initializer: {kernel_initializer}")


def kernel_initializer_wrapper(x):
    if x is None:
        return x
    if isinstance(x, str):
        return prepare_kernel_initializers(x)
    assert callable(x)
    return x


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


class VarianceScaling:
    def __init__(self, scale=0.1, mode="fan_in", distribution="uniform"):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

        assert mode == "fan_in" or mode == "fan_out" or mode == "fan_avg"
        assert distribution == "uniform", "only uniform distribution is supported"

    def __call__(self, tensor: Tensor) -> None:
        with torch.no_grad():
            scale = self.scale
            shape = tensor.shape
            in_units, out_units = calc_units(shape)
            if self.mode == "fan_in":
                scale /= in_units
            elif self.mode == "fan_out":
                scale /= out_units
            else:
                scale /= (in_units + out_units) / 2

            limit = math.sqrt(3.0 * scale)
            return tensor.uniform_(-limit, limit)


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


def _normalize_hidden_state(S: torch.Tensor) -> torch.Tensor:
    """Normalizes the hidden state tensor as described in the paper."""
    # (B, *)
    # Handles both (B, W) for dense and (B, C, H, W) for spatial

    if S.dim() == 2:
        # Case: (B, W) - Dense layer output
        S_norm = S
        dim = 1
    elif S.dim() == 4:
        # Case: (B, C, H, W) - Spatial output. Normalize over spatial dimensions (H*W) for each channel.
        # Reshape to (B*C, H*W) or (B, C, H*W)
        B, C, H, W = S.shape
        S_norm = S.view(B, C, H * W)  # (B, C, H*W)
        dim = 2  # Normalize across H*W dimension for each channel
    else:
        # Not a standard MuZero shape, return unnormalized to be safe
        return S

    min_hidden_state = S_norm.min(dim=dim, keepdim=True)[0]
    max_hidden_state = S_norm.max(dim=dim, keepdim=True)[0]

    if S.dim() == 4:
        # For spatial normalization, restore original dimensions after min/max
        min_hidden_state = min_hidden_state.unsqueeze(-1)  # (B, C, 1, 1)
        max_hidden_state = max_hidden_state.unsqueeze(-1)

    scale_hidden_state = max_hidden_state - min_hidden_state

    # Handle the case where min == max
    scale_hidden_state[scale_hidden_state < 1e-5] = (
        1.0  # Or use += 1e-5 if that's the original intent
    )

    hidden_state = (S - min_hidden_state) / scale_hidden_state
    return hidden_state


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


@dataclass
class NetworkOutput:
    """
    Represents the Agent's Hypothesis (Predictions).
    Shape: (B, Unroll+1, ...)
    """

    # Core MuZero
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor

    # Latents (Needed for Consistency)
    latent_state: torch.Tensor  # s_k
    projected_state: torch.Tensor = None  # For EfficientZero consistency

    # Stochastic MuZero
    chance_logits: torch.Tensor = None
    afterstate_value: torch.Tensor = None
    encoder_codes: torch.Tensor = None  # One-hot from Encoder
    encoder_probs: torch.Tensor = None  # Softmax from Encoder


# Helper function to use across your framework
def get_clean_state_dict(model: torch.nn.Module) -> dict:
    """Strips the '_orig_mod.' prefix added by torch.compile."""
    state_dict = model.state_dict()
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


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
            "search_afterstate",
            "step",
        ]
        for attr in compiled_methods:
            if attr in m_copy.__dict__:
                del m_copy.__dict__[attr]

        return m_copy

    return model
