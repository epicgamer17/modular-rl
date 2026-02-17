from typing import Callable, Tuple

from configs.agents.muzero import MuZeroConfig
from torch import nn, Tensor
from modules.action_encoder import ActionEncoder
from modules.network_block import NetworkBlock
from modules.backbones.factory import BackboneFactory
from modules.sim_siam_projector_predictor import Projector
from modules.heads.to_play import ToPlayHead
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.reward import RewardHead
from modules.output_strategy_factory import OutputStrategyFactory
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


class Prediction(nn.Module):
    def __init__(
        self,
        backbone_config,
        value_config,
        policy_config,
        output_size: int,
        input_shape: Tuple[int],
        arch_config,
        game_is_discrete: bool = True,
    ):
        super().__init__()

        self.net = BackboneFactory.create(backbone_config, input_shape)

        # Value
        val_strategy = OutputStrategyFactory.create(value_config.output_strategy)
        self.value_head = ValueHead(
            arch_config=arch_config,
            input_shape=self.net.output_shape,
            strategy=val_strategy,
            neck_config=value_config.neck,
        )

        # Policy (or Chance)
        pol_strategy = OutputStrategyFactory.create(policy_config.output_strategy)
        self.policy_head = PolicyHead(
            arch_config=arch_config,
            input_shape=self.net.output_shape,
            num_actions=output_size,
            neck_config=policy_config.neck,
            strategy=pol_strategy,
        )

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)
        self.value_head.initialize()
        self.policy_head.initialize()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S = self.net(inputs)
        return self.value_head(S), self.policy_head(S)


class Encoder(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_codes: int = 32,
    ):
        """
        Args:
            config: MuZeroConfig containing chance_encoder_backbone.
            input_shape: tuple, e.g. (C, H, W) or (B, C, H, W).
            num_codes: embedding size output by encoder.
        """
        super().__init__()
        self.config = config
        self.num_codes = num_codes

        # Use modular backbone for Encoder
        self.net = BackboneFactory.create(config.chance_encoder_backbone, input_shape)

        # Output head: maps backbone output to num_codes
        backbone_output_shape = self.net.output_shape
        # Output head: maps backbone output to num_codes
        backbone_output_shape = self.net.output_shape
        flat_dim = 1
        for d in backbone_output_shape:
            flat_dim *= d

        self.fc = nn.Linear(flat_dim, num_codes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            probs: (B, num_codes) - Softmax probabilities
            one_hot_st: (B, num_codes) - Straight-Through gradient flow
        """
        # 1. Processing to Logits
        x = self.net(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)
        x = self.fc(x)

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
        self.net.initialize(initializer)
        zero_weights_initializer(self.fc)


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

        self.prediction = Prediction(
            backbone_config=config.prediction_backbone,
            value_config=config.value_head,
            policy_config=config.policy_head,
            output_size=num_actions,
            input_shape=hidden_state_shape,
            arch_config=config.arch,
            game_is_discrete=config.game.is_discrete,
        )
        if self.config.stochastic:
            self.afterstate_prediction = Prediction(
                backbone_config=config.prediction_backbone,
                value_config=config.value_head,
                policy_config=config.chance_probability_head,
                output_size=config.num_chance,
                input_shape=hidden_state_shape,
                arch_config=config.arch,
                game_is_discrete=config.game.is_discrete,
            )

        # --- 4. EFFICIENT ZERO Projector ---
        # The flat hidden dimension is simply the total size of the hidden state
        self.flat_hidden_dim = torch.Size(hidden_state_shape).numel()
        self.projector = Projector(self.flat_hidden_dim, config)

        encoder_input_shape = list(input_shape)
        # Input is (C, H, W) or (D,)
        # We concatenate two observations, so channels/feature dim doubles
        encoder_input_shape[0] = input_shape[0] * 2
        encoder_input_shape = tuple(encoder_input_shape)
        print("encoder input shape", encoder_input_shape)
        self.encoder = Encoder(
            config,
            encoder_input_shape,
            num_codes=self.config.num_chance,
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
