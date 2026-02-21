from typing import Callable, Tuple, Dict, List, Optional, Any
from modules.agent_nets.base import BaseAgentNetwork

from configs.agents.muzero import MuZeroConfig
from torch import nn, Tensor
import torch
import torch.nn.functional as F


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
    PhysicsOutput,
    WorldModelOutput,
)
from utils.utils import to_lists, normalize_images, recursive_batch

from modules.blocks.conv import Conv2dStack
from modules.blocks.dense import DenseStack, build_dense
from modules.blocks.residual import ResidualStack


class MuZeroNetwork(BaseAgentNetwork):
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
        super().__init__()
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

    def batch_network_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        return recursive_batch(states)

    def unbatch_network_states(
        self, batched_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        from utils.utils import recursive_unbatch

        return recursive_unbatch(batched_state)

    @torch.inference_mode()
    def obs_inference(self, obs: Tensor) -> "InferenceOutput":
        """
        Actor API: translates a raw observation into an initial latent state,
        value estimate, and action distribution for use by ActionSelectors or
        MCTS root expansion.

        Args:
            obs: Observation tensor. May be unbatched (shape == input_shape)
                 or batched (shape == (B, *input_shape)).

        Returns:
            InferenceOutput with:
                - ``network_state``: Opaque dict ``{"dynamics": hidden_state,
                  "wm_memory": wm_head_state}`` for MCTS recurrent steps.
                - ``value``: Expected value scalar(s) V(s).
                - ``policy``: Action distribution (Categorical etc.).
                - ``reward``: None (no reward at the initial step).
                - ``to_play``: Turn indicator (for multi-player games).
        """
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Add batch dimension if the input is a single (unbatched) observation.
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        # Representation network: obs → initial hidden state
        wm_output = self.world_model.initial_inference(obs)
        hidden_state = wm_output.features

        # Prediction backbone + heads
        pred_features = self.prediction_backbone(hidden_state)
        raw_value, _ = self.value_head(pred_features)
        raw_policy, _, policy_dist = self.policy_head(pred_features)

        expected_value = self.value_head.strategy.to_expected_value(raw_value)

        network_state = {
            "dynamics": hidden_state,
            "wm_memory": wm_output.head_state,
        }

        return InferenceOutput(
            network_state=network_state,
            value=expected_value,
            policy=policy_dist,
            reward=None,  # No reward at the root step
            to_play=wm_output.to_play,
            extras={},
        )

    @torch.inference_mode()
    def hidden_state_inference(
        self,
        network_state: Dict[str, Tensor | dict],
        action: Tensor,
    ) -> InferenceOutput:
        """
        MCTS Search API: steps the latent dynamics model (state, action) →
        (next_state, reward, value, policy).

        Called at every simulated node expansion inside MCTS. The ``network_state``
        token is treated as an opaque object — its internal structure is packed and
        unpacked here, invisible to the search tree.

        Args:
            network_state: Opaque dict produced by the previous ``obs_inference``
                           or ``hidden_state_inference`` call.
            action: One-hot or integer action tensor of shape ``(B, num_actions)``
                    or ``(B, 1)``.

        Returns:
            InferenceOutput with updated ``network_state``, ``value``, ``policy``,
            and ``reward`` for this simulated transition.
        """
        dynamics_state = network_state["dynamics"]
        wm_memory = network_state.get("wm_memory")

        wm_output: WorldModelOutput = self.world_model.recurrent_inference(
            hidden_state=dynamics_state,
            action=action,
            recurrent_state=wm_memory,
        )

        next_hidden = wm_output.features
        pred_features = self.prediction_backbone(next_hidden)

        raw_value, _ = self.value_head(pred_features)
        _, _, policy_dist = self.policy_head(pred_features)

        expected_value = self.value_head.strategy.to_expected_value(raw_value)

        next_network_state = {
            "dynamics": next_hidden,
            "wm_memory": wm_output.head_state,
        }

        return InferenceOutput(
            network_state=next_network_state,
            value=expected_value,
            policy=policy_dist,
            reward=wm_output.instant_reward,
            to_play=wm_output.to_play,
            extras={},
        )

    @torch.inference_mode()
    def afterstate_inference(
        self, network_state: Any, action: Tensor
    ) -> InferenceOutput:
        """
        Stochastic MCTS Search API: steps the latent model to the intermediate
        afterstate (post-action, pre-environment-chance).

        Used by Stochastic MuZero to model environment stochasticity.

        Args:
            network_state: Opaque dict from the previous search step.
            action: Action tensor.

        Returns:
            InferenceOutput with the afterstate ``network_state``, afterstate
            ``value``, and ``policy`` / ``chance`` distributions over chance codes.
        """
        wm_output = self.world_model.afterstate_recurrent_inference(
            network_state, action
        )
        shared_features = wm_output.features
        chance_logits = wm_output.chance

        raw_value, _ = self.afterstate_value_head(shared_features)

        network_state_after = {
            "dynamics": wm_output.afterstate_features,
            "wm_memory": network_state["wm_memory"],
        }

        return InferenceOutput(
            network_state=network_state_after,
            value=self.afterstate_value_head.strategy.to_expected_value(raw_value),
            policy=self.world_model.sigma_head.strategy.get_distribution(chance_logits),
            chance=self.world_model.sigma_head.strategy.get_distribution(chance_logits),
            reward=None,
        )

    def learner_inference(
        self,
        batch: Any,
    ) -> "LearningOutput":
        """
        Learner API: unrolls the world model and runs all prediction heads,
        returning raw logits for the loss pipeline.

        This method satisfies the ``BaseAgentNetwork`` learner contract. All
        outputs are strongly typed via ``LearningOutput``; the MuZero-specific
        fields (latents, afterstates, chance codes, etc.) map to named fields
        rather than an opaque extras dict.

        Args:
            batch: Dict containing at minimum:
                - ``"observations"``: Float tensor ``(B, *input_shape)``.
                - ``"actions"``: Long tensor ``(B, T)`` of unroll actions.
                - ``"unroll_observations"`` (optional): ``(B, T+1, *input_shape)``
                  for EfficientZero consistency targets.
                - ``"chance_codes"`` (optional): ``(B, T)`` for Stochastic MuZero.

        Returns:
            LearningOutput with tensors of shape ``(B, T+1, ...)``:
                - ``values``, ``policies``, ``rewards``, ``to_plays``.
                - ``latents``: Latent states (for consistency loss).
                - ``latents_afterstates``: Afterstate latents (stochastic only).
                - ``chance_logits``: Chance code logits (stochastic only).
                - ``chance_values``: Afterstate value predictions (stochastic only).
                - ``chance_encoder_softmaxes``: Encoder softmax outputs (stochastic only).
        """
        from modules.world_models.inference_output import LearningOutput

        initial_observation = batch["observations"]
        actions = batch["actions"]
        target_observations = batch.get("unroll_observations")
        target_chance_codes = batch.get("chance_codes")

        assert initial_observation is not None, "Batch must contain 'observations'"
        assert actions is not None, "Batch must contain 'actions'"

        # 1. Initial representation: obs → latent
        wm_output = self.world_model.initial_inference(initial_observation)
        latent = wm_output.features
        head_state = wm_output.head_state

        # 2. Build encoder inputs for Stochastic MuZero (pairs of consecutive obs)
        encoder_inputs = None
        if self.config.stochastic and target_observations is not None:
            # target_observations: [B, T+1, *obs_shape]
            # Concatenate neighbouring pairs along the channel dim for the encoder
            encoder_inputs = torch.cat(
                [target_observations[:, :-1], target_observations[:, 1:]], dim=2
            )

        # 3. World model unroll: latent → full sequence of latents + aux outputs
        physics_output = self.world_model.unroll_physics(
            initial_latent_state=latent,
            actions=actions,
            encoder_inputs=encoder_inputs,
            true_chance_codes=target_chance_codes,
            head_state=head_state,
            target_observations=target_observations,
        )

        # 4. Prediction heads over the full unrolled sequence
        stacked_latents = physics_output.latents  # [B, T+1, *latent_shape]
        B, T_plus_1 = stacked_latents.shape[:2]
        flat_latents = stacked_latents.reshape(B * T_plus_1, *stacked_latents.shape[2:])

        pred_features = self.prediction_backbone(flat_latents)
        raw_values, _ = self.value_head(pred_features)  # [B*T+1, ...]
        raw_policies, _, _ = self.policy_head(pred_features)  # [B*T+1, num_actions]

        # Reshape back to sequence form: [B, T+1, ...]
        raw_values = raw_values.view(B, T_plus_1, -1)
        raw_policies = raw_policies.view(B, T_plus_1, -1)

        # 5. Stochastic-MuZero afterstate heads
        latents_afterstates: Optional[Tensor] = None
        stochastic_chance_logits: Optional[Tensor] = None
        stochastic_chance_values: Optional[Tensor] = None
        stochastic_encoder_softmaxes: Optional[Tensor] = None

        if self.config.stochastic and physics_output.latents_afterstates is not None:
            latents_afterstates = physics_output.latents_afterstates

            stacked_backbone_features = physics_output.afterstate_backbone_features
            B_as, T_plus_1_as = stacked_backbone_features.shape[:2]
            flat_backbone = stacked_backbone_features.view(
                B_as * T_plus_1_as, *stacked_backbone_features.shape[2:]
            )

            raw_chance_values, _ = self.afterstate_value_head(flat_backbone)
            stochastic_chance_values = raw_chance_values.view(B_as, T_plus_1_as, -1)
            stochastic_chance_logits = physics_output.chance_logits
            stochastic_encoder_softmaxes = physics_output.chance_encoder_softmaxes

        return LearningOutput(
            values=raw_values,
            policies=raw_policies,
            rewards=physics_output.rewards,
            to_plays=physics_output.to_plays,
            latents=stacked_latents,
            latents_afterstates=latents_afterstates,
            chance_logits=stochastic_chance_logits,
            chance_values=stochastic_chance_values,
            chance_encoder_softmaxes=stochastic_encoder_softmaxes,
        )

    def project(self, hidden_state: Tensor, grad=True) -> Tensor:
        """
        Projects the hidden state (s_t) into the embedding space.
        Used for both the 'real' target observation and the 'predicted' latent.
        """
        # Save original shape to restore later
        original_shape = hidden_state.shape
        # Flatten the spatial/latent dimensions (C, H, W) or (H,)
        # Ensure we keep the leading batch/sequence dimensions
        flat_hidden = hidden_state.reshape(-1, self.flat_hidden_dim)
        proj = self.projector.projection(flat_hidden)

        # with grad, use proj_head
        if grad:
            proj = self.projector.projection_head(proj)
        else:
            proj = proj.detach()

        # Restore leading shape: (B, T, embedding_dim) or (B, embedding_dim)
        # We replace the latent_shape part with embedding_dim
        # hidden_state_shape could be (C, H, W) (3 dims) or (H,) (1 dim)
        hidden_state_shape = self.world_model.representation.output_shape
        num_latent_dims = len(hidden_state_shape)
        new_shape = list(original_shape[:-num_latent_dims]) + [proj.shape[-1]]
        return proj.reshape(new_shape)
