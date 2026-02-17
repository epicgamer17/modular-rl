from typing import Callable, List, Optional, Tuple, Dict

from torch import Tensor
import torch
from modules.embeddings.action_embedding import ActionEncoder
from modules.blocks.dense import build_dense
from modules.backbones.factory import BackboneFactory
from modules.heads.factory import HeadFactory
from modules.heads.to_play import ToPlayHead
from modules.heads.reward import RewardHead
from modules.heads.strategy_factory import OutputStrategyFactory
from modules.utils import _normalize_hidden_state, scale_gradient
from modules.world_models.world_model import WorldModelInterface
from configs.agents.muzero import MuZeroConfig

from torch import nn
import torch.nn.functional as F

from modules.world_models.world_model import WorldModelOutput


# --- Refactored Primary Modules ---
class Representation(nn.Module):
    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int]):
        super().__init__()
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"
        self.config = config
        self.net = BackboneFactory.create(config.representation_backbone, input_shape)
        self.output_shape = self.net.output_shape

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        S = self.net(inputs)
        # Apply normalization to the final output of the representation network
        return _normalize_hidden_state(S)


class BaseDynamics(nn.Module):
    """Base class for Dynamics and AfterstateDynamics, handling action fusion and core block."""

    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
        layer_prefix: str,
    ):
        super().__init__()
        self.config = config
        self.action_embedding_dim = action_embedding_dim
        is_continuous = not self.config.game.is_discrete

        # 1. Action Encoder
        self.action_encoder = ActionEncoder(
            action_space_size=num_actions,
            embedding_dim=self.action_embedding_dim,
            is_continuous=is_continuous,
            single_action_plane=(
                layer_prefix == "dynamics"
            ),  # Assuming standard dynamics uses single_action_plane=True
            # TODO: FIX THIS AND DONT MAKE THIS ASSUMPTION
        )

        # 2. Fusion Layer (Move from ActionEncoder to Dynamics)
        if len(input_shape) == 3:
            # Image input (C, H, W)
            self.num_channels = input_shape[0]
            in_channels = self.num_channels + self.action_embedding_dim
            out_channels = self.num_channels
            self.fusion = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.fusion_bn = nn.BatchNorm2d(out_channels)
        else:
            # Vector input (C,) or (D,)
            self.input_size = input_shape[0]
            in_features = self.input_size + self.action_embedding_dim
            out_features = self.input_size
            self.fusion = nn.Linear(in_features, out_features, bias=False)
            self.fusion_bn = nn.BatchNorm1d(out_features)

        # 3. Core Network Block
        if layer_prefix == "dynamics":
            bb_cfg = config.dynamics_backbone
        elif layer_prefix == "afterstate_dynamics":
            bb_cfg = config.afterstate_dynamics_backbone

        self.net = BackboneFactory.create(bb_cfg, input_shape)
        self.output_shape = self.net.output_shape

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)
        # Additional initializations for fusion layers if needed

    def _fuse_and_process(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # Embed action
        action_embedded = self.action_encoder(action, hidden_state.shape)

        # Concatenate and fuse
        x = torch.cat((hidden_state, action_embedded), dim=1)
        x = self.fusion(x)
        # x = self.fusion_bn(x) # BN is often omitted or placed after ReLU in some MuZero implementations

        # Residual Connection
        x = x + hidden_state
        S = F.relu(x)

        # Process through the main network block
        S = self.net(S)

        # Apply normalization to the final output of the dynamics network
        next_hidden_state = _normalize_hidden_state(S)

        return next_hidden_state


class Dynamics(BaseDynamics):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
    ):
        # dynamics layers uses the "dynamics" prefix
        super().__init__(
            config, input_shape, num_actions, action_embedding_dim, "dynamics"
        )
        # 3. Heads
        # Reward Head
        r_strategy = OutputStrategyFactory.create(config.reward_head.output_strategy)
        self.reward_head = HeadFactory.create(
            config.reward_head,
            config.arch,
            input_shape=self.output_shape,
            strategy=r_strategy,
        )

        # To-Play Head
        tp_strategy = OutputStrategyFactory.create(config.to_play_head.output_strategy)
        self.to_play_head = ToPlayHead(
            arch_config=config.arch,
            input_shape=self.output_shape,
            num_players=config.game.num_players,
            neck_config=config.to_play_head.neck,
            strategy=tp_strategy,
        )

        # LSTM Support for Value Prefix is now handled inside ValuePrefixRewardHead
        # if config.reward_head is a ValuePrefixRewardHeadConfig.

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        super().initialize(initializer)
        self.reward_head.initialize()
        self.to_play_head.initialize()

    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        reward_hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:

        next_hidden_state = self._fuse_and_process(hidden_state, action)

        # Predict reward (and handle state if ValuePrefixRewardHead)
        # We pass the full state dict, knowing RewardHead will look for "reward_hidden"
        # and return a new state dict with "reward_hidden" if updated.
        state = {"reward_hidden": reward_hidden}
        reward, new_state = self.reward_head(next_hidden_state, state=state)

        # Extract new reward hidden state or use the old one if not updated
        new_reward_hidden = new_state.get("reward_hidden", reward_hidden)

        # To Play
        to_play, _ = self.to_play_head(next_hidden_state)
        return reward, next_hidden_state, to_play, new_reward_hidden


class AfterstateDynamics(BaseDynamics):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
    ):
        super().__init__(
            config,
            input_shape,
            num_actions,
            action_embedding_dim,
            "afterstate_dynamics",
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The base class handles fusion and processing, returning the normalized hidden state (afterstate)
        afterstate = self._fuse_and_process(hidden_state, action)
        return afterstate


from modules.world_models.components.chance_encoder import ChanceEncoder
from modules.world_models.world_model import WorldModelOutput


class MuzeroWorldModel(WorldModelInterface, nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        observation_dimensions: Tuple[int, ...],
        num_actions: int,
    ):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        # 1. Representation Network
        self.representation = Representation(config, observation_dimensions)
        self.num_chance = config.num_chance

        hidden_state_shape = self.representation.output_shape

        # --- 3. Dynamics and Prediction Networks ---
        if self.config.stochastic:
            self.afterstate_dynamics = AfterstateDynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
            )

            # Stochastic Dynamics(Afterstate, Code) -> Next State
            self.dynamics = Dynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_chance,
                action_embedding_dim=self.config.action_embedding_dim,
            )

            # Shared Backbone for Afterstate Features (Sigma and Value Heads)
            self.shared_backbone = BackboneFactory.create(
                self.config.prediction_backbone, self.representation.output_shape
            )

            # Sigma Head (Owned by WorldModel - fundamental environment rules)
            self.sigma_head = HeadFactory.create(
                self.config.chance_probability_head,
                self.config.arch,
                input_shape=self.shared_backbone.output_shape,
                num_chance_codes=self.num_chance,
            )

            # Encoder (Moves from AgentNetwork to WorldModel)
            encoder_input_shape = list(observation_dimensions)
            encoder_input_shape[0] *= 2  # Double channels for stacked obs
            self.encoder = ChanceEncoder(
                config,
                tuple(encoder_input_shape),
                num_codes=self.num_chance,
            )

        else:
            self.dynamics = Dynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
            )

        # Dynamics output must match Representation output shape
        assert (
            self.dynamics.output_shape == self.representation.output_shape
        ), f"{self.dynamics.output_shape} = {self.representation.output_shape}"

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.representation.initialize(initializer)
        self.dynamics.initialize(initializer)
        if hasattr(self, "afterstate_dynamics"):
            self.afterstate_dynamics.initialize(initializer)
        if hasattr(self, "shared_backbone"):
            self.shared_backbone.initialize(initializer)
        if hasattr(self, "sigma_head"):
            self.sigma_head.initialize(initializer)
        if hasattr(self, "encoder"):
            self.encoder.initialize(initializer)

    def initial_inference(self, observation: Tensor) -> WorldModelOutput:
        hidden_state = self.representation(observation)
        return WorldModelOutput(features=hidden_state)

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        reward_h_states: Optional[Tensor],
        reward_c_states: Optional[Tensor],
    ) -> WorldModelOutput:
        if not self.config.stochastic:
            action = action.view(-1).to(hidden_state.device)
            # one-hot the action -> (B, num_actions)
            action = (
                F.one_hot(action.long(), num_classes=self.num_actions)
                .float()
                .to(hidden_state.device)
            )

        reward, next_hidden_state, to_play, reward_hidden = self.dynamics(
            hidden_state, action, (reward_h_states, reward_c_states)
        )

        return WorldModelOutput(
            features=next_hidden_state,
            reward=reward,
            to_play=to_play,
            reward_hidden=reward_hidden,
        )

    def afterstate_recurrent_inference(
        self,
        hidden_state,
        action,
    ) -> WorldModelOutput:
        # 1. Transition to Afterstate
        action = action.view(-1).to(hidden_state.device)
        # one-hot the action -> (B, num_actions)
        action = (
            F.one_hot(action.long(), num_classes=self.num_actions)
            .float()
            .to(hidden_state.device)
        )

        afterstate_latent = self.afterstate_dynamics(hidden_state, action)

        # 2. Extract Shared Features (Backbone)
        shared_features = self.shared_backbone(afterstate_latent)

        # 3. Predict Chance Probabilities (Logits)
        chance_logits, _ = self.sigma_head(shared_features)

        return WorldModelOutput(
            afterstate_features=afterstate_latent,  # Raw dynamics output
            features=shared_features,  # Shared processed features for Agent
            chance=chance_logits,  # Environment chance rules
        )

    def unroll_sequence(
        self,
        initial_hidden_state: Tensor,
        actions: Tensor,
        target_observations: Tensor,
        target_chance_codes: Tensor,
        reward_h_states: Tensor,
        reward_c_states: Tensor,
        preprocess_fn: Callable[[Tensor], Tensor],
    ) -> Dict[str, List[Tensor]]:
        """
        Unrolls the physics engine (Representation -> Dynamics -> Reward).
        Does NOT compute Policy or Value.
        Returns a dictionary containing sequences of latent states, rewards, etc.
        """
        # --- 3. Initialize Storage Lists ---
        hidden_states = initial_hidden_state
        latent_states = [hidden_states]  # length will end up being unroll_steps + 1

        # Stochastic MuZero specific storage
        if self.config.stochastic:
            latent_afterstates = [
                torch.zeros_like(hidden_states).to(hidden_states.device)
            ]
            latent_code_probabilities = [
                torch.zeros(
                    (self.config.minibatch_size, self.config.num_chance),
                    device=initial_hidden_state.device,
                )
            ]
            encoder_softmaxes = [
                torch.zeros(
                    (self.config.minibatch_size, self.config.num_chance),
                    device=initial_hidden_state.device,
                )
            ]
            encoder_onehots = [
                torch.zeros(
                    (self.config.minibatch_size, self.config.num_chance),
                    device=initial_hidden_state.device,
                )
            ]
        else:
            latent_afterstates = []
            latent_code_probabilities = []
            encoder_softmaxes = []
            encoder_onehots = []

        if self.config.support_range is not None:
            reward_shape = (
                self.config.minibatch_size,
                self.config.support_range * 2 + 1,
            )
        else:
            reward_shape = (self.config.minibatch_size, 1)

        rewards = [
            torch.zeros(reward_shape, device=initial_hidden_state.device)
        ]  # R_t = 0 (Placeholder)

        to_plays = [
            torch.zeros(
                (self.config.minibatch_size, self.config.game.num_players),
                device=initial_hidden_state.device,
            )
        ]

        # --- 4. Unroll Loop ---
        for k in range(self.config.unroll_steps):
            actions_k = actions[:, k]

            if self.config.stochastic:
                target_observations_k = target_observations[:, k]
                target_observations_k_plus_1 = target_observations[:, k + 1]
                real_obs_k = preprocess_fn(target_observations_k)
                real_obs_k_plus_1 = preprocess_fn(target_observations_k_plus_1)
                encoder_input = torch.concat([real_obs_k, real_obs_k_plus_1], dim=1)

                # 1. Afterstate Inference
                afterstate_out = self.afterstate_recurrent_inference(
                    hidden_states, actions_k
                )
                afterstates = afterstate_out.afterstate_features
                shared_features = afterstate_out.features
                chance_logits_k = afterstate_out.chance

                # 3. Encoder Inference
                encoder_softmax_k, encoder_onehot_k = self.encoder(encoder_input)

                if self.config.use_true_chance_codes:
                    codes_k = F.one_hot(
                        target_chance_codes[:, k + 1].squeeze(-1).long(),
                        self.config.num_chance,
                    )
                    encoder_onehot_k = codes_k.float()

                latent_afterstates.append(afterstates)
                latent_code_probabilities.append(chance_logits_k)  # Logits

                # Store encoder outputs
                encoder_onehots.append(encoder_onehot_k)
                encoder_softmaxes.append(encoder_softmax_k)

                # 4. Dynamics Inference (using chance code as action)
                reward, next_hidden_state, to_play, reward_hidden = self.dynamics(
                    afterstates, encoder_onehot_k, (reward_h_states, reward_c_states)
                )

                rewards_k = reward
                hidden_states = next_hidden_state
                to_plays_k = to_play

                # Update LSTM states
                reward_h_states = reward_hidden[0]
                reward_c_states = reward_hidden[1]

            else:
                # Standard MuZero
                wm_output = self.recurrent_inference(
                    hidden_states,
                    actions_k,
                    reward_h_states,
                    reward_c_states,
                )
                rewards_k = wm_output.reward
                hidden_states = wm_output.features
                to_plays_k = wm_output.to_play

                reward_h_states = wm_output.reward_hidden[0]
                reward_c_states = wm_output.reward_hidden[1]

            latent_states.append(hidden_states)
            rewards.append(rewards_k)
            to_plays.append(to_plays_k)

            # Scale the gradient of the hidden state (applies to the whole batch)
            hidden_states = scale_gradient(hidden_states, 0.5)

            # reset hidden states
            if self.config.value_prefix and (k + 1) % self.config.lstm_horizon_len == 0:
                reward_h_states = torch.zeros_like(reward_h_states).to(
                    hidden_states.device
                )
                reward_c_states = torch.zeros_like(reward_c_states).to(
                    hidden_states.device
                )

        return {
            "rewards": rewards,
            "to_plays": to_plays,
            "latent_states": latent_states,
            "latent_afterstates": latent_afterstates,
            # "latent_code_probabilities": latent_code_probabilities, # These are agent outputs, calculated by Composer?
            "encoder_softmaxes": encoder_softmaxes,
            "encoder_onehots": encoder_onehots,
            # "chance_values": chance_values, # Agent output
        }

    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            "representation_network": self.representation,
            "dynamics_network": self.dynamics,
            "afterstate_dynamics_network": getattr(self, "afterstate_dynamics", None),
        }
