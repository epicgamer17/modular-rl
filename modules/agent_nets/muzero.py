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
from modules.world_models.inference_output import InferenceOutput, UnrollOutput
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

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.world_model.initialize(initializer)

        self.prediction_backbone.initialize(initializer)
        self.value_head.initialize(initializer)
        self.policy_head.initialize(initializer)

        if self.config.stochastic:
            self.afterstate_value_head.initialize(initializer)

        # Initialize projector?

    @torch.no_grad()
    def initial_inference(self, obs: Tensor) -> InferenceOutput:
        """Actor/MCTS API: Translates latent states into expected values."""

        # Ask World Model for physics state
        wm_output = self.world_model.initial_inference(obs)
        hidden_state = wm_output.features

        # Pass hidden state through shared prediction backbone
        pred_features = self.prediction_backbone(hidden_state)

        # Ask Agent Heads for opinions
        raw_value, _ = self.value_head(pred_features)
        raw_policy, _ = self.policy_head(pred_features)

        # Translate to MCTS-ready math
        return InferenceOutput(
            network_state=hidden_state,
            value=self.value_head.strategy.to_expected_value(raw_value),
            policy=self.policy_head.strategy.get_distribution(raw_policy),
            reward=None,  # Initial inference has no reward
            chance=None,
            to_play=None,  # Initial inference doesn't usually return to_play, or does it?
            # Root to_play is usually from environment.
        )

    @torch.no_grad()
    def recurrent_inference(
        self,
        network_state: Dict[str, Tensor],
        action: Tensor,
    ) -> InferenceOutput:

        # 1. Unpack opaque state
        hidden_state = network_state["dynamics"]
        reward_hidden = network_state.get("reward_hidden")

        # Ask World Model for next physics state
        wm_output = self.world_model.recurrent_inference(
            hidden_state, action, reward_hidden
        )

        next_hidden = wm_output.features
        raw_reward = wm_output.reward
        raw_to_play = wm_output.to_play

        # Pass next hidden state through shared prediction backbone
        pred_features = self.prediction_backbone(next_hidden)

        # Ask Agent Heads
        raw_value, _ = self.value_head(pred_features)
        raw_policy, _ = self.policy_head(pred_features)

        # Translate
        # Note: reward and to_play come from WorldModel (Dynamics), so we need their strategies too.
        # But strategies are attached to heads inside WorldModel?
        # Yes, WorldModel owns RewardHead. We access it via wm.dynamics.reward_head.strategy?
        # Or we duplicate the strategy logic here?
        # Ideally WorldModel heads allow access to strategy.

        expected_value = self.value_head.strategy.to_expected_value(raw_value)
        expected_reward = (
            self.world_model.dynamics.reward_head.strategy.to_expected_value(raw_reward)
        )

        # To Play
        # to_play is usually categorical probs or logits.
        to_play_dist = self.world_model.dynamics.to_play_head.strategy.get_distribution(
            raw_to_play
        )
        # MCTS expects to_play as int usually? Or distribution?
        # The new InferenceOutput has to_play as Optional[int | Tensor].
        # Let's return the most likely player for now or the distribution?
        # ModularSearch likely wants the integer to flip nodes.
        # But let's check modular_search.py usage. It uses argmax.

        return InferenceOutput(
            network_state=next_hidden,  # This might need to be wrapped if we want symmetry? But caller will wrap it or MCTS handles it?
            # MCTS expects "network_state" to be the hidden_state usually, but wait.
            # If I pass `network_state={"dynamics": ...}` in, I should probably return a similar structure/contract?
            # The previous implementation returned `network_state=next_hidden` and `extras={"reward_hidden": ...}`.
            # If I want to be opaque, `network_state` in InferenceOutput should probably act as the unique state descriptor.
            # usage in MCTS: `hidden_state = outputs.network_state`.
            # And `reward_hidden = outputs.extras["reward_hidden"]`.
            # If I change MCTS to use `outputs.network_state["dynamics"]`, that would be cleaner divergence.
            # But let's look at `initial_inference`.
            value=expected_value,
            policy=self.policy_head.strategy.get_distribution(raw_policy),
            reward=expected_reward,
            to_play=raw_to_play,
            extras={"reward_hidden": wm_output.reward_hidden},
        )

    @torch.no_grad()
    def afterstate_recurrent_inference(self, hidden_state, action) -> InferenceOutput:
        # Ask World Model for afterstate and shared features
        wm_output = self.world_model.afterstate_recurrent_inference(
            hidden_state, action
        )
        shared_features = wm_output.features
        chance_logits = wm_output.chance

        # Piggyback on shared features for value prediction
        raw_value, _ = self.afterstate_value_head(shared_features)

        return InferenceOutput(
            network_state=wm_output.afterstate_features,
            value=self.afterstate_value_head.strategy.to_expected_value(raw_value),
            policy=self.world_model.sigma_head.strategy.get_distribution(chance_logits),
            chance=self.world_model.sigma_head.strategy.get_distribution(chance_logits),
            reward=None,
        )

    def unroll_sequence(
        self,
        initial_hidden_state: Tensor,
        initial_values: Tensor,  # Unused?
        initial_policies: Tensor,  # Unused?
        actions: Tensor,
        target_observations: Tensor,
        target_chance_codes: Tensor,
        reward_h_states: Tensor,
        reward_c_states: Tensor,
        preprocess_fn: Callable[[Tensor], Tensor],
    ) -> UnrollOutput:
        """Learner API: Returns pure logits for loss computation."""

        # 1. Unroll Physics
        physics_output = self.world_model.unroll_sequence(
            initial_hidden_state=initial_hidden_state,
            actions=actions,
            target_observations=target_observations,
            target_chance_codes=target_chance_codes,
            reward_h_states=reward_h_states,
            reward_c_states=reward_c_states,
            preprocess_fn=preprocess_fn,
        )

        latent_states = physics_output["latent_states"]
        # List of Tensors. Stack them.
        # latent_states is [hidden_0, hidden_1, ... hidden_K]
        stacked_latents = torch.stack(latent_states, dim=1)  # (B, K+1, ...)

        # 2. Pass sequence through Agent Heads
        # We can crush time dimension to batch dimension for efficiency
        B, T = stacked_latents.shape[:2]
        flat_latents = stacked_latents.reshape(B * T, *stacked_latents.shape[2:])

        # Shared prediction backbone
        pred_features = self.prediction_backbone(flat_latents)

        raw_values, _ = self.value_head(pred_features)
        raw_policies, _ = self.policy_head(pred_features)

        # Uncrush
        raw_values = raw_values.view(B, T, -1)
        raw_policies = raw_policies.view(B, T, -1)

        # Rewards and ToPlays are already lists of tensors from unroll_physics
        stacked_rewards = (
            torch.stack(physics_output["rewards"], dim=1)
            if len(physics_output["rewards"]) > 0
            else torch.empty(0)
        )
        stacked_toplays = (
            torch.stack(physics_output["to_plays"], dim=1)
            if len(physics_output["to_plays"]) > 0
            else torch.empty(0)
        )

        # Handle Stochastic Extra Outputs
        chance_values = None
        chance_logits = None
        latents_afterstates = None

        if self.config.stochastic:
            # 1. Stack Afterstates
            stacked_afterstates = torch.stack(
                physics_output["latent_afterstates"], dim=1
            )
            latents_afterstates = stacked_afterstates

            # 2. Extract Shared Features (Backbone output stored in chance_out dictionary for convenience or re-computed?)
            # Currently physics_output doesn't return stacked shared features.
            # We can re-run the backbone or update unroll_physics to return them.
            # Re-running is safer for now if we didn't store them.

            B_as, T_as = stacked_afterstates.shape[:2]
            flat_afterstates = stacked_afterstates.view(
                B_as * T_as, *stacked_afterstates.shape[2:]
            )

            shared_features = self.world_model.shared_backbone(flat_afterstates)

            # 3. Predict Afterstate Values (Agent Objective)
            raw_chance_values, _ = self.afterstate_value_head(shared_features)
            chance_values = raw_chance_values.view(B_as, T_as, -1)

            # 4. Predict Chance Logits (Environment Rules - piggyback)
            raw_chance_logits, _ = self.world_model.sigma_head(shared_features)
            chance_logits = raw_chance_logits.view(B_as, T_as, -1)

        return UnrollOutput(
            values=raw_values,
            policies=raw_policies,
            rewards=stacked_rewards,
            to_plays=stacked_toplays,
            latents=stacked_latents,
            latents_afterstates=latents_afterstates,
            chance_logits=chance_logits,
            chance_values=chance_values,
            extras=physics_output,  # Still pass physics output for encoder stats/onehots
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
