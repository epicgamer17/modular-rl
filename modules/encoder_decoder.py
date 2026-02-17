from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as td

from modules.blocks.conv import Conv2dStack, ConvTranspose2dStack
from modules.blocks.dense import DenseStack
from modules.distributions import TanhBijector, SampleDist


class ConvEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int], depth=32, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.depth = depth

        # Hardcoded architecture from reference: 4 layers, stride 2
        # Input: (B, C, H, W) or (C, H, W)
        # We use Conv2dStack for consistency
        self.net = Conv2dStack(
            input_shape=input_shape,
            filters=[1 * depth, 2 * depth, 4 * depth, 8 * depth],
            kernel_sizes=[4, 4, 4, 4],
            strides=[2, 2, 2, 2],
            activation=activation,
            norm_type="none",  # Dreamer often doesn't use BN in encoder
        )
        self.output_dim = 32 * depth  # Flattened dimension assumption for 64x64 input

    def forward(self, obs: Tensor) -> Tensor:
        # obs input might need reshaping if it comes as dict or different format
        # Assuming obs is tensor (B, C, H, W)
        x = self.net(obs)
        return x.reshape(x.shape[0], -1)


class ConvDecoder(nn.Module):
    def __init__(self, feat_size, depth=32, shape=(3, 64, 64), activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.depth = depth
        self.shape = shape  # (C, H, W)

        self.linear = nn.Linear(feat_size, 32 * depth)

        # Transpose Conv Stack
        # Logic to upscale 1x1 -> 64x64
        # We need to manually define this to match the TF implementation logic

        self.decoder_net = ConvTranspose2dStack(
            input_shape=(32 * depth, 1, 1),  # Dummy shape for initialization
            filters=[4 * depth, 2 * depth, 1 * depth, shape[0]],
            kernel_sizes=[5, 5, 6, 6],
            strides=[2, 2, 2, 2],
            activation=activation,
            norm_type="none",
            # The last layer usually shouldn't have activation if it outputs mean
            # but Stack applies to all. We handle last layer activation manually/identity.
        )

    def forward(self, features: Tensor) -> td.Distribution:
        x = self.linear(features)
        x = x.reshape(x.shape[0], 32 * self.depth, 1, 1)

        # Manual forward through stack to control final activation
        for i, layer in enumerate(self.decoder_net._layers):
            x = layer(x)
            # Apply activation to all except the very last one (Image Mean)
            if i < len(self.decoder_net._layers) - 1:
                x = self.activation(x)

        mean = x
        return td.Independent(td.Normal(mean, 1.0), 3)


class ActionDecoder(nn.Module):
    def __init__(
        self,
        feat_size: int,
        action_size: int,
        layers: int = 4,
        units: int = 400,
        dist: str = "tanh_normal",
        activation=nn.ELU(),
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
    ):
        super().__init__()
        self._size = action_size
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        # Use DenseStack
        widths = [units] * layers
        self.net = DenseStack(
            initial_width=feat_size,
            widths=widths,
            activation=activation,
            norm_type="none",
        )

        if dist == "tanh_normal":
            self.head = nn.Linear(self.net.output_width, 2 * action_size)
        elif dist == "onehot":
            self.head = nn.Linear(self.net.output_width, action_size)

    def forward(self, features: Tensor):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = self.net(features)
        x = self.head(x)

        if self._dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + raw_init_std) + self._min_std

            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, TanhBijector())
            dist = td.Independent(dist, 1)
            return SampleDist(dist)  # Custom wrapper for mode/sample

        elif self._dist == "onehot":
            return td.OneHotCategorical(logits=x)


class DenseDecoder(nn.Module):
    def __init__(
        self, feat_size, shape, layers, units, dist="normal", activation=nn.ELU()
    ):
        super().__init__()
        self._shape = shape
        self._dist = dist

        widths = [units] * layers
        self.net = DenseStack(
            initial_width=feat_size, widths=widths, activation=activation
        )
        self.head = nn.Linear(self.net.output_width, int(np.prod(shape)))

    def forward(self, features):
        x = self.net(features)
        x = self.head(x)
        x = x.reshape(x.shape[0], *self._shape)

        if self._dist == "normal":
            return td.Independent(td.Normal(x, 1.0), len(self._shape))
        if self._dist == "binary":
            return td.Independent(td.Bernoulli(logits=x), len(self._shape))
