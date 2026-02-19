from typing import Callable, Tuple, Dict, List, Optional, Any

from configs.agents.muzero import MuZeroConfig
from torch import nn, Tensor
import torch
import torch.nn.functional as F


from modules.network_block import NetworkBlock
from modules.backbones.factory import BackboneFactory

from modules.projectors.sim_siam import Projector
from modules.heads.to_play import ToPlayHead
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.reward import RewardHead
from modules.heads.strategy_factory import OutputStrategyFactory
from modules.utils import _normalize_hidden_state, zero_weights_initializer
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.world_models.inference_output import (
    InferenceOutput,
    UnrollOutput,
    PhysicsOutput,
    WorldModelOutput,
)
from utils.utils import to_lists

from modules.blocks.conv import Conv2dStack
from modules.blocks.dense import DenseStack, build_dense
from modules.blocks.residual import ResidualStack


class AgentNetwork(nn.Module):
    """
    The Composer: Wires a Physics Engine (WorldModel) to an Agent's Objectives (Heads).
    """

    def __init__(
        self,
        config: MuZeroConfig,
        num_actions: int,
        input_shape: Tuple[int],
        channel_first: bool = True,
        world_model_cls=MuzeroWorldModel,
    ):
        super(AgentNetwork, self).__init__()
        self.config = config
        self.channel_first = channel_first
        self.num_actions = num_actions
        self.input_shape = input_shape

        # 1. The Physics Engine
        self.world_model = world_model_cls(config, input_shape, num_actions)

        hidden_state_shape = self.world_model.representation.output_shape

        # 2. The Task-Specific Heads
        # Restore shared prediction backbone
        self.prediction_backbone = BackboneFactory.create(
            config.prediction_backbone, hidden_state_shape
        )
        prediction_feat_shape = self.prediction_backbone.output_shape

        # Value
        val_strategy = OutputStrategyFactory.create(config.value_head.output_strategy)
        self.value_head = ValueHead(
            arch_config=config.arch,
            input_shape=prediction_feat_shape,
            strategy=val_strategy,
            neck_config=config.value_head.neck,
        )

        # Policy
        pol_strategy = OutputStrategyFactory.create(config.policy_head.output_strategy)
        self.policy_head = PolicyHead(
            arch_config=config.arch,
            input_shape=prediction_feat_shape,
            neck_config=config.policy_head.neck,
            strategy=pol_strategy,
        )

        # Stochastic Chance Heads (if applicable)
        if self.config.stochastic:
            # Afterstate Value Head (Piggybacks on shared backbone in World Model)
            shared_backbone_output_shape = self.world_model.shared_backbone.output_shape
            self.afterstate_value_head = ValueHead(
                arch_config=config.arch,
                input_shape=shared_backbone_output_shape,
                strategy=val_strategy,
                neck_config=config.value_head.neck,
            )

        # --- 4. EFFICIENT ZERO Projector ---
        # The flat hidden dimension is simply the total size of the hidden state
        self.flat_hidden_dim = torch.Size(hidden_state_shape).numel()
        self.projector = Projector(self.flat_hidden_dim, config)

    @property
    def device(self) -> torch.device:
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.world_model.initialize(initializer)

        self.prediction_backbone.initialize(initializer)
        self.value_head.initialize(initializer)
        self.policy_head.initialize(initializer)

        if self.config.stochastic:
            self.afterstate_value_head.initialize(initializer)

        # Initialize projector?

    def batch_reward_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        from utils.utils import recursive_batch

        return recursive_batch(states)

    def unbatch_reward_states(
        self, batched_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        from utils.utils import recursive_unbatch

        return recursive_unbatch(batched_state)

    @torch.no_grad()
    def initial_inference(self, obs: Tensor) -> "InferenceOutput":
        """Actor/MCTS API: Translates latent states into expected values."""

        # Ensure obs is a tensor
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Ensure obs has batch dim
        # Assuming input_shape from config does NOT include batch
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        # Ask World Model for physics state
        wm_output = self.world_model.initial_inference(obs)
        hidden_state = wm_output.features

        # Pass through prediction backbone
        pred_features = self.prediction_backbone(hidden_state)

        # Ask Agent Heads
        raw_value, _ = self.value_head(pred_features)
        raw_policy, _, policy_dist = self.policy_head(pred_features)

        # Translate
        expected_value = self.value_head.strategy.to_expected_value(raw_value)
        # policy_dist is already computed by PolicyHead

        # Initial Reward State
        reward_hidden = self.world_model.reward_head.get_initial_state(
            batch_size=obs.shape[0], device=obs.device
        )

        # Wrap everything in a unique opaque state
        network_state = {
            "dynamics": hidden_state,
            "heads": reward_hidden,
        }

        return InferenceOutput(
            network_state=network_state,
            value=expected_value,
            policy=policy_dist,
            policy_logits=raw_policy,
            reward=None,  # Initial inference has no reward
            to_play=wm_output.to_play,
            extras={},
        )

    @torch.no_grad()
    def recurrent_inference(
        self,
        network_state: Dict[str, Tensor | dict],
        action: Tensor,
    ) -> InferenceOutput:

        # 1. Unpack opaque state
        dynamics_state = network_state["dynamics"]
        reward_state = network_state["heads"]

        # Use world model dynamics and heads
        wm_output: WorldModelOutput = self.world_model.recurrent_inference(
            hidden_state=dynamics_state,
            action=action,
            recurrent_state=reward_state,
        )

        # TRUST THE HEAD: The RewardHead now returns the instant reward directly,
        # handling its own value prefix subtraction and horizon reset internally.
        instant_reward = wm_output.instant_reward

        next_hidden = wm_output.features
        raw_to_play = wm_output.to_play
        next_reward_state = wm_output.head_state
        # reward_hidden = reward_state.get("reward_hidden")  # No longer needed to extract internally here if we trust the heads state through shared prediction backbone
        pred_features = self.prediction_backbone(next_hidden)

        # Ask Agent Heads
        raw_value, _ = self.value_head(pred_features)
        raw_policy, _, policy_dist = self.policy_head(pred_features)

        # Translate
        expected_value = self.value_head.strategy.to_expected_value(raw_value)

        # To Play
        to_play_dist = self.world_model.to_play_head.strategy.get_distribution(
            raw_to_play
        )

        next_network_state = {
            "dynamics": next_hidden,
            "heads": next_reward_state,
        }

        return InferenceOutput(
            network_state=next_network_state,
            value=expected_value,
            policy=policy_dist,
            policy_logits=raw_policy,
            reward=instant_reward,
            to_play=raw_to_play,
            extras={},
        )

    @torch.no_grad()
    def afterstate_recurrent_inference(
        self, network_state: Any, action
    ) -> InferenceOutput:
        # Ask World Model for afterstate and shared features
        wm_output = self.world_model.afterstate_recurrent_inference(
            network_state, action
        )
        shared_features = wm_output.features
        chance_logits = wm_output.chance

        # Piggyback on shared features for value prediction
        raw_value, _ = self.afterstate_value_head(shared_features)

        # Wrap everything in a unique opaque state
        network_state_after = {
            "dynamics": wm_output.afterstate_features,
            "heads": network_state["heads"],
        }

        return InferenceOutput(
            network_state=network_state_after,
            value=self.afterstate_value_head.strategy.to_expected_value(raw_value),
            policy=self.world_model.sigma_head.strategy.get_distribution(chance_logits),
            chance=self.world_model.sigma_head.strategy.get_distribution(chance_logits),
            reward=None,
        )

    def unroll_sequence(
        self,
        initial_network_state: Any,
        actions: Tensor,
        target_observations: Tensor,
        target_chance_codes: Tensor,
        preprocess_fn: Callable[[Tensor], Tensor],
    ) -> UnrollOutput:
        """Learner API: Returns pure logits for loss computation."""

        # 1. Unpack Opaque State if provided
        latent = initial_network_state["dynamics"]
        head_state = initial_network_state.get("heads")
        # 1. Unroll Physics
        physics_output = self.world_model.unroll_physics(  # Changed to unroll_physics
            initial_latent_state=latent,
            actions=actions,
            target_observations=target_observations,
            target_chance_codes=target_chance_codes,
            initial_reward_state=head_state,
            preprocess_fn=preprocess_fn,
        )

        # PhysicsOutput now contains STACKED tensors
        stacked_latents = physics_output.latents

        # 2. Pass sequence through Agent Heads
        # We can crush time dimension to batch dimension for efficiency
        B, T = stacked_latents.shape[:2]
        flat_latents = stacked_latents.reshape(B * T, *stacked_latents.shape[2:])

        # Shared prediction backbone
        pred_features = self.prediction_backbone(flat_latents)

        raw_values, _ = self.value_head(pred_features)
        raw_policies, _, _ = self.policy_head(pred_features)

        # Uncrush
        raw_values = raw_values.view(B, T, -1)
        raw_policies = raw_policies.view(B, T, -1)

        # Rewards and ToPlays are already stacked tensors from unroll_physics
        stacked_rewards = physics_output.rewards
        stacked_toplays = physics_output.to_plays

        # Handle Stochastic Extra Outputs
        chance_values = None
        chance_logits = None
        latents_afterstates = None

        if self.config.stochastic and physics_output.latents_afterstates is not None:
            # 1. Stack Afterstates (Dynamics Output) - Already stacked
            stacked_afterstates = physics_output.latents_afterstates
            latents_afterstates = stacked_afterstates

            # 2. Extract Shared Backbone Features (Already computed in physics loop) - Already stacked
            stacked_backbone_features = physics_output.afterstate_backbone_features

            B_as, T_as = stacked_backbone_features.shape[:2]
            flat_backbone = stacked_backbone_features.view(
                B_as * T_as, *stacked_backbone_features.shape[2:]
            )

            # 3. Predict Afterstate Values
            raw_chance_values, _ = self.afterstate_value_head(flat_backbone)
            chance_values = raw_chance_values.view(B_as, T_as, -1)

            # 4. Use Chance Logits directly from physics loop
            chance_logits = physics_output.chance_logits

        # Create extras dict for stochastic outputs
        extras = {}
        if physics_output.encoder_softmaxes is not None:
            extras["encoder_softmaxes"] = physics_output.encoder_softmaxes
        if physics_output.encoder_onehots is not None:
            extras["encoder_onehots"] = physics_output.encoder_onehots

        return UnrollOutput(
            values=raw_values,
            policies=raw_policies,
            rewards=stacked_rewards,
            to_plays=stacked_toplays,
            latents=stacked_latents,
            latents_afterstates=latents_afterstates,
            chance_logits=chance_logits,
            chance_values=chance_values,
            extras=extras if extras else None,
        )

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
