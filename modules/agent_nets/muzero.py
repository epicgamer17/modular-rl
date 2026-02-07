from typing import Callable, Tuple

from agent_configs.muzero_config import MuZeroConfig
from torch import nn, Tensor
from modules.action_encoder import ActionEncoder
from modules.actor import ActorNetwork
from modules.critic import CriticNetwork
from modules.network_block import NetworkBlock
from modules.sim_siam_projector_predictor import Projector
from modules.utils import _normalize_hidden_state, zero_weights_initializer
from modules.world_models.muzero_world_model import MuzeroWorldModel
from utils.utils import to_lists

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from modules.residual import ResidualStack
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    """
    Combines the final network block output with the Actor and Critic heads.
    This replaces the 'prediction' logic from the old Prediction class forward.
    """

    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int], output_size: int):
        super().__init__()
        self.critic = CriticNetwork(config, input_shape)
        self.actor = ActorNetwork(config, input_shape, output_size)

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.actor.initialize(initializer)
        self.critic.initialize(initializer)

    def forward(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic(S), self.actor(S)


class Prediction(nn.Module):
    def __init__(self, config: MuZeroConfig, output_size: int, input_shape: Tuple[int]):
        super().__init__()
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"
        self.config = config

        self.net = NetworkBlock(
            config, input_shape
        )  # Uses default layers (config.residual_layers etc.)
        self.head = PredictionHead(config, self.net.output_shape, output_size)

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)
        self.head.initialize(initializer)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S = self.net(inputs)
        return self.head(S)


class Encoder(nn.Module):
    def __init__(
        self,
        input_shape,
        num_codes: int = 32,
        channel_first: bool = True,
    ):
        """
        Args:
            input_shape: tuple, e.g. (C, H, W) or (B, C, H, W).
            num_codes: embedding size output by encoder.
            channel_first: if True, treats input as (..., C, H, W).
        """
        super().__init__()
        self.channel_first = channel_first
        self.num_codes = num_codes

        # --- Image / 4D Path ---
        if len(input_shape) == 4:
            # Heuristic: treating len=4 or len=3 (C,H,W) as image data
            self.use_conv = True

            # Determine input channels based on flag
            # Assuming input_shape excludes Batch dim if len==3, or includes it if len==4
            if len(input_shape) == 4:
                c = input_shape[1] if self.channel_first else input_shape[3]
            else:
                c = input_shape[0] if self.channel_first else input_shape[2]

            self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1)
            self.act = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

            # --- REMOVED POOLING HERE ---
            # previously: self.pool = nn.AdaptiveAvgPool2d(1)

            # We must calculate the flattened size dynamically because
            # we no longer force the output to 1x1.
            with torch.no_grad():
                # Create a dummy input to trace shape
                # We assume input_shape provided matches the user's tensor setup
                if len(input_shape) == 3:
                    dummy_in = torch.zeros(1, *input_shape)
                else:
                    dummy_in = torch.zeros(*input_shape)

                # Check for channel last input and permute for PyTorch convs if necessary
                if not self.channel_first and dummy_in.shape[-1] == c:
                    dummy_in = dummy_in.permute(0, 3, 1, 2)

                d = self.conv1(dummy_in)
                d = self.conv2(d)
                d = self.conv3(d)
                flat_dim = d.flatten(1).shape[1]

            # Linear head: maps (64 * H' * W') -> num_codes
            self.fc = nn.Linear(flat_dim, num_codes)

        # --- Vector / 2D Path ---
        else:
            self.use_conv = False
            # Assuming input_shape is (Batch, Features) or just (Features,)
            self.fc1 = nn.Linear(input_shape[-1], 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_codes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            probs: (B, num_codes) - Softmax probabilities
            one_hot_st: (B, num_codes) - Straight-Through gradient flow
        """
        # 1. Processing to Logits
        if self.use_conv:
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))

            # Flatten directly (preserving spatial info in the flat vector)
            x = x.flatten(1, -1)
            x = self.fc(x)  # (B, num_codes)
        else:
            # Vector path returns LOGITS (removed .softmax inside here to avoid double softmax)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

        # 2. Softmax
        probs = x.softmax(dim=-1)

        # Convert to one-hot (B, num_codes)
        one_hot = torch.zeros_like(probs).scatter_(
            -1, torch.argmax(probs, dim=-1, keepdim=True), 1.0
        )

        # # 4. Straight-Through Estimator
        # # Forward: use one_hot
        # # Backward: use gradients of probs
        one_hot_st = (one_hot - probs).detach() + probs

        # TODO: LIKE LIGHT ZERO NO SOFTMAX?
        # probs = x
        # one_hot_st = OnehotArgmax.apply(probs)

        # one_hot_st = F.gumbel_softmax(x, tau=1.0, hard=True, dim=-1)
        # probs = F.softmax(x, dim=-1)
        # TODO: output logits or probs here?
        probs = x
        return probs, one_hot_st

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        if self.use_conv:
            initializer(self.conv1.weight)
            initializer(self.conv2.weight)
            initializer(self.conv3.weight)
            # zero initializer for fc
            zero_weights_initializer(self.fc)
        else:
            initializer(self.fc1.weight)
            initializer(self.fc2.weight)
            zero_weights_initializer(self.fc3)


class OnehotArgmax(torch.autograd.Function):
    """
    Overview:
        Custom PyTorch function for one-hot argmax. This function transforms the input tensor \
        into a one-hot tensor where the index with the maximum value in the original tensor is \
        set to 1 and all other indices are set to 0. It allows gradients to flow to the encoder \
        during backpropagation.

        For more information, refer to: \
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input):
        """
        Overview:
            Forward method for the one-hot argmax function. This method transforms the input \
            tensor into a one-hot tensor.
        Arguments:
            - ctx (:obj:`context`): A context object that can be used to stash information for
            backward computation.
            - input (:obj:`torch.Tensor`): Input tensor.
        Returns:
            - (:obj:`torch.Tensor`): One-hot tensor.
        """
        # Transform the input tensor to a one-hot tensor
        return torch.zeros_like(input).scatter_(
            -1, torch.argmax(input, dim=-1, keepdim=True), 1.0
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Overview:
            Backward method for the one-hot argmax function. This method allows gradients \
            to flow to the encoder during backpropagation.
        Arguments:
            - ctx (:obj:`context`):  A context object that was stashed in the forward pass.
            - grad_output (:obj:`torch.Tensor`): The gradient of the output tensor.
        Returns:
            - (:obj:`torch.Tensor`): The gradient of the input tensor.
        """
        return grad_output


class Network(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        num_actions: int,
        input_shape: Tuple[int],
        channel_first: bool = True,
        world_model_cls=MuzeroWorldModel,
    ):
        super(Network, self).__init__()
        self.config = config
        self.channel_first = channel_first

        self.world_model = world_model_cls(config, input_shape, num_actions)

        hidden_state_shape = self.world_model.representation.output_shape
        print("Hidden state shape:", hidden_state_shape)

        self.prediction = Prediction(config, num_actions, hidden_state_shape)
        if self.config.stochastic:
            self.afterstate_prediction = Prediction(
                config, config.num_chance, hidden_state_shape
            )

        # --- 4. EFFICIENT ZERO Projector ---
        # The flat hidden dimension is simply the total size of the hidden state
        self.flat_hidden_dim = torch.Size(hidden_state_shape[1:]).numel()
        self.projector = Projector(self.flat_hidden_dim, config)

        encoder_input_shape = list(input_shape)
        encoder_input_shape[1] = input_shape[1] * 2
        encoder_input_shape = tuple(encoder_input_shape)
        print("encoder input shape", encoder_input_shape)
        self.encoder = Encoder(
            encoder_input_shape,
            num_codes=self.config.num_chance,
            channel_first=channel_first,
        )

    def initial_inference(self, obs):
        wm_output = self.world_model.initial_inference(obs)
        hidden_state = wm_output.features
        value, policy = self.prediction(hidden_state)
        return value, policy, hidden_state

    def recurrent_inference(
        self,
        hidden_state,
        action,
        reward_h_states,
        reward_c_states,
    ):
        wm_output = self.world_model.recurrent_inference(
            hidden_state, action, reward_h_states, reward_c_states
        )

        reward = wm_output.reward
        next_hidden_state = wm_output.features
        to_play = wm_output.to_play
        reward_hidden = wm_output.reward_hidden
        value, policy = self.prediction(next_hidden_state)
        return reward, next_hidden_state, value, policy, to_play, reward_hidden

    def afterstate_recurrent_inference(
        self,
        hidden_state,
        action,
    ):
        wm_output = self.world_model.afterstate_recurrent_inference(
            hidden_state, action
        )

        afterstate = wm_output.afterstate_features
        value, sigma = self.afterstate_prediction(afterstate)
        return afterstate, value, sigma

    def project(self, hidden_state: Tensor, grad=True) -> Tensor:
        """
        Projects the hidden state (s_t) into the embedding space.
        Used for both the 'real' target observation and the 'predicted' latent.
        """
        # Flatten the spatial dimensions (B, C, H, W) -> (B, C*H*W)
        flat_hidden = hidden_state.flatten(1, -1)
        proj = self.projector.projection(flat_hidden)

        # with grad, use proj_head
        if grad:
            proj = self.projector.projection_head(proj)
            return proj
        else:
            return proj.detach()
