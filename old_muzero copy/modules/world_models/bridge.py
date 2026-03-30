"""
WorldModelBridge: adapts new ActionFusion/SpatialEmbedding into the old MuzeroWorldModel interface.

Used by the A/B comparison notebook (Cell 2) to test the new action embedding approach
in isolation, leaving all other components (Representation, Dynamics backbone, heads) unchanged.

What is NEW (under test):
- ActionFusion with SpatialActionEmbedding for board games (e.g. TicTacToe)

What is OLD (unchanged for isolation):
- Representation network
- Dynamics backbone
- RewardHead + ToPlayHead
- unroll_physics loop logic
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Any, Tuple, Optional, Dict

from old_muzero.modules.world_models.inference_output import WorldModelOutput, PhysicsOutput
from old_muzero.modules.world_models.world_model import WorldModelInterface


class WorldModelBridge(WorldModelInterface, nn.Module):
    """Wraps NEW ActionFusion/SpatialEmbedding in the OLD MuzeroWorldModel interface.

    Constructor matches old MuzeroWorldModel: (config, observation_dimensions, num_actions).
    ModularAgentNetwork._init_muzero can drop this in via the world_model_cls kwarg.
    """

    def __init__(self, config: Any, observation_dimensions: Tuple[int, ...], num_actions: int):
        super().__init__()
        self.config = config
        self.num_actions = num_actions

        # 1. OLD Representation — provides .output_shape and obs->latent encoding
        from old_muzero.modules.world_models.components.representation import Representation
        self.representation = Representation(config, observation_dimensions)
        latent_shape = self.representation.output_shape

        # 2. NEW ActionFusion — selects embedding based on environment type
        from modules.embeddings.action_embedding import ActionEncoder
        from modules.embeddings.action_fusion import ActionFusion

        is_spatial = len(latent_shape) == 3
        if is_spatial:
            h, w = latent_shape[1], latent_shape[2]
            if num_actions == h * w:
                from modules.embeddings.actions.spatial import SpatialActionEmbedding
                emb = SpatialActionEmbedding(num_actions, h, w)
                emb_dim = 1
                print(f"  WorldModelBridge: SpatialActionEmbedding ({h}x{w} board)")
            else:
                from modules.embeddings.actions.efficient_zero import EfficientZeroActionEmbedding
                emb = EfficientZeroActionEmbedding(num_actions, config.action_embedding_dim)
                emb_dim = config.action_embedding_dim
                print(f"  WorldModelBridge: EfficientZeroActionEmbedding (dim={emb_dim})")
        else:
            from modules.embeddings.actions.efficient_zero import EfficientZeroActionEmbedding
            emb = EfficientZeroActionEmbedding(num_actions, config.action_embedding_dim)
            emb_dim = config.action_embedding_dim
            print(f"  WorldModelBridge: EfficientZeroActionEmbedding (vector, dim={emb_dim})")

        action_encoder = ActionEncoder(emb, emb_dim)
        self.action_fusion = ActionFusion(
            encoder=action_encoder, input_shape=latent_shape, use_bn=True
        )

        # 3. OLD Dynamics backbone
        from old_muzero.modules.backbones.factory import BackboneFactory
        self.dynamics_net = BackboneFactory.create(config.dynamics_backbone, latent_shape)
        assert self.dynamics_net.output_shape == latent_shape, (
            f"Dynamics output {self.dynamics_net.output_shape} != latent {latent_shape}"
        )

        # 4. OLD Reward head
        from old_muzero.modules.heads.factory import HeadFactory
        from old_muzero.agents.learner.losses.representations import get_representation
        r_rep = get_representation(config.reward_head.output_strategy)
        self.reward_head = HeadFactory.create(
            config.reward_head, config.arch,
            input_shape=self.dynamics_net.output_shape, representation=r_rep,
        )

        # 5. OLD ToPlay head
        from old_muzero.modules.heads.to_play import ToPlayHead
        tp_rep = get_representation(config.to_play_head.output_strategy)
        self.to_play_head = ToPlayHead(
            arch_config=config.arch,
            input_shape=self.dynamics_net.output_shape,
            num_players=config.game.num_players,
            neck_config=config.to_play_head.neck,
            representation=tp_rep,
        )

    def _forward_dynamics(self, latent: Tensor, action_onehot: Tensor) -> Tensor:
        """NEW ActionFusion -> OLD backbone -> normalize."""
        from old_muzero.modules.utils import _normalize_hidden_state
        fused = self.action_fusion(latent, action_onehot)
        next_latent = self.dynamics_net(fused)
        return _normalize_hidden_state(next_latent)

    def initial_inference(self, observation: Tensor) -> WorldModelOutput:
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(observation, dtype=torch.float32)
        if observation.dim() == len(self.representation.input_shape):
            observation = observation.unsqueeze(0)
        hidden_state = self.representation(observation.float())
        return WorldModelOutput(features=hidden_state)

    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor, recurrent_state: Any = None
    ) -> WorldModelOutput:
        action_oh = F.one_hot(action.view(-1).long(), num_classes=self.num_actions).float()
        action_oh = action_oh.to(hidden_state.device)
        next_latent = self._forward_dynamics(hidden_state, action_oh)
        reward_logits, new_state, instant_reward = self.reward_head(
            next_latent, state=recurrent_state
        )
        to_play_logits, _, to_play = self.to_play_head(next_latent)
        return WorldModelOutput(
            features=next_latent,
            reward=reward_logits,
            to_play=to_play,
            to_play_logits=to_play_logits,
            head_state=new_state,
            instant_reward=instant_reward,
        )

    def unroll_physics(
        self,
        initial_latent_state: Tensor,
        actions: Tensor,
        encoder_inputs: Optional[Tensor] = None,
        true_chance_codes: Optional[Tensor] = None,
        head_state: Any = None,
        target_observations: Optional[Tensor] = None,
    ) -> PhysicsOutput:
        from old_muzero.modules.utils import scale_gradient

        latent_states = [initial_latent_state]
        rewards: list[Tensor] = []
        to_plays: list[Tensor] = []

        # Root to_play prediction (index 0, matching old MuzeroWorldModel convention)
        initial_to_play_logits, _, _ = self.to_play_head(initial_latent_state)
        to_plays.append(initial_to_play_logits)

        current_latent = initial_latent_state
        current_head_state = head_state

        for k in range(actions.shape[1]):
            wm_out = self.recurrent_inference(current_latent, actions[:, k], current_head_state)
            rewards.append(wm_out.reward)
            to_plays.append(wm_out.to_play_logits)
            current_latent = wm_out.features
            current_head_state = wm_out.head_state
            latent_states.append(current_latent)
            current_latent = scale_gradient(current_latent, 0.5)

        stacked_latents = torch.stack(latent_states, dim=1)
        stacked_rewards = torch.stack(rewards, dim=1) if rewards else torch.empty(0)
        stacked_to_plays = torch.stack(to_plays, dim=1)

        stacked_target_latents = None
        if target_observations is not None and self.config.consistency_loss_factor > 0:
            B, T1 = target_observations.shape[:2]
            flat = target_observations.reshape(B * T1, *target_observations.shape[2:])
            with torch.no_grad():
                tl = self.representation(flat.float())
                stacked_target_latents = tl.view(B, T1, *tl.shape[1:])

        return PhysicsOutput(
            latents=stacked_latents,
            rewards=stacked_rewards,
            to_plays=stacked_to_plays,
            target_latents=stacked_target_latents,
        )

    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            "representation_network": self.representation,
            "dynamics_network": self.dynamics_net,
        }
