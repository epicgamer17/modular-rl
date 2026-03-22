from typing import List, Optional, Tuple, Dict, Any, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from configs.agents.muzero import MuZeroConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.conditioned import ConditionedBackbone
from modules.heads.factory import HeadFactory
from modules.utils import scale_gradient
from modules.models.inference_output import WorldModelOutput


class WorldModel(nn.Module):
    """
    A modular world model that encapsulates the representation, dynamics, 
    and environment heads. Everything that extracts features is a Backbone. 
    Everything that predicts semantics is a Head.
    """
    def __init__(
        self,
        config: MuZeroConfig,
        observation_dimensions: Tuple[int, ...],
        num_actions: int,
    ):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_chance = config.num_chance
        
        # 1. Representation Network (Pure observation backbone)
        self.representation = BackboneFactory.create(
            config.representation_backbone, observation_dimensions
        )

        # 2. Dynamics Networks (Action-conditioned backbones)
        if self.config.stochastic:
            # Afterstate Dynamics: (Latent, Action) -> Afterstate
            self.afterstate_dynamics = ConditionedBackbone(
                config=self.config,
                input_shape=self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
                backbone_config=self.config.afterstate_dynamics_backbone,
            )

            # Dynamics: (Afterstate, Chance) -> Next Latent
            self.dynamics = ConditionedBackbone(
                config=self.config,
                input_shape=self.representation.output_shape,
                num_actions=self.num_chance,
                action_embedding_dim=self.config.action_embedding_dim,
                backbone_config=self.config.dynamics_backbone,
            )

            # Prediction backbone (for chance probability)
            self.prediction_backbone = BackboneFactory.create(
                self.config.prediction_backbone, self.representation.output_shape
            )

            self.sigma_head = HeadFactory.create(
                self.config.chance_probability_head,
                self.config.arch,
                input_shape=self.prediction_backbone.output_shape,
                num_chance_codes=self.num_chance,
            )

            # Chance Encoder: (Stacked Obs) -> Chance Codes
            encoder_input_shape = list(observation_dimensions)
            encoder_input_shape[0] *= 2 # Double channels for stacked obs
            self.encoder = BackboneFactory.create(
                config.chance_encoder_backbone, tuple(encoder_input_shape)
            )
            # Map flattened backbone output to codes
            flat_dim = 1
            for d in self.encoder.output_shape: flat_dim *= d
            self.chance_projector = nn.Linear(flat_dim, self.num_chance)
        else:
            # Deterministic Dynamics: (Latent, Action) -> Next Latent
            self.dynamics = ConditionedBackbone(
                config=self.config,
                input_shape=self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
                backbone_config=self.config.dynamics_backbone,
            )

        # 3. Environment Heads (Physics Engine)
        self.heads = nn.ModuleDict()
        
        from agents.learner.losses.representations import get_representation
        if config.reward_head is not None:
            r_rep = get_representation(config.reward_head.output_strategy)
            self.heads["reward_logits"] = HeadFactory.create(
                config.reward_head,
                config.arch,
                input_shape=self.dynamics.output_shape,
                representation=r_rep,
            )

        if config.continuation_head is not None:
            c_rep = get_representation(config.continuation_head.output_strategy)
            self.heads["continuation_logits"] = HeadFactory.create(
                config.continuation_head,
                config.arch,
                input_shape=self.dynamics.output_shape,
                representation=c_rep,
            )

        if config.to_play_head is not None:
            tp_rep = get_representation(config.to_play_head.output_strategy)
            self.heads["to_play_logits"] = HeadFactory.create(
                config.to_play_head,
                config.arch,
                input_shape=self.dynamics.output_shape,
                num_players=config.game.num_players,
                representation=tp_rep,
            )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def initial_inference(self, observation: Tensor) -> WorldModelOutput:
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if observation.dim() == len(self.representation.input_shape):
            observation = observation.unsqueeze(0)

        hidden_state = self.representation(observation.float())
        return WorldModelOutput(features=hidden_state)

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        recurrent_state: Any = None,
    ) -> WorldModelOutput:
        # Action is expected as one-hot for stochastic dynamics, or single-integer for deterministic
        next_hidden_state = self.dynamics(hidden_state, action)
        
        predictions = {}
        head_state = {} if recurrent_state is None else recurrent_state
        new_head_state = {}

        for name, head in self.heads.items():
            h_state = head_state.get(name) if isinstance(head_state, dict) else None
            logits, n_state, extra = head(next_hidden_state, state=h_state)
            predictions[name] = logits
            predictions[f"{name}_extra"] = extra
            new_head_state[name] = n_state

        return WorldModelOutput(
            features=next_hidden_state,
            reward=predictions.get("reward_logits"),
            to_play_logits=predictions.get("to_play_logits"),
            to_play=predictions.get("to_play_logits_extra"),
            continuation_logits=predictions.get("continuation_logits"),
            continuation=predictions.get("continuation_logits_extra"),
            head_state=new_head_state,
            instant_reward=predictions.get("reward_logits_extra"),
        )

    def afterstate_recurrent_inference(
        self,
        network_state: Dict[str, Any],
        action: Tensor,
    ) -> WorldModelOutput:
        latent_state = network_state["dynamics"]
        
        afterstate_latent = self.afterstate_dynamics(latent_state, action)
        shared_features = self.prediction_backbone(afterstate_latent)
        chance_logits, _, _ = self.sigma_head(shared_features)

        return WorldModelOutput(
            afterstate_features=afterstate_latent,
            features=shared_features,
            chance=chance_logits,
        )

    def unroll_physics(
        self,
        initial_latent_state: Tensor,
        actions: Tensor,
        encoder_inputs: Optional[Tensor] = None,
        true_chance_codes: Optional[Tensor] = None,
        head_state: Any = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        unroll_steps = actions.shape[1]
        latents = [initial_latent_state]
        head_sequences = {name: [] for name in self.heads.keys()}
        
        current_latent = initial_latent_state
        current_head_state = head_state if head_state is not None else {}
        
        # Initial head prediction for root
        for name, head in self.heads.items():
            h_state = current_head_state.get(name) if isinstance(current_head_state, dict) else None
            logits, n_state, _ = head(current_latent, state=h_state)
            head_sequences[name].append(logits)
            if isinstance(current_head_state, dict):
                current_head_state[name] = n_state

        stochastic_sequences = {
            "latents_afterstates": [],
            "chance_logits": [],
            "afterstate_backbone_features": [],
        } if self.config.stochastic else {}

        for k in range(unroll_steps):
            action_k = actions[:, k]

            if self.config.stochastic:
                afterstate_out = self.afterstate_recurrent_inference(
                    {"dynamics": current_latent}, action_k
                )
                afterstate = afterstate_out.afterstate_features
                shared_features = afterstate_out.features
                chance_logits = afterstate_out.chance

                if self.config.use_true_chance_codes and true_chance_codes is not None:
                    # [B, 1] -> [B, NumChance]
                    codes_k = F.one_hot(
                        true_chance_codes[:, k + 1].squeeze(-1).long(),
                        self.num_chance,
                    ).float()
                else:
                    # Chance Encoder logic: ST-estimator for chance codes
                    x = self.encoder(encoder_inputs[:, k])
                    x = x.flatten(1, -1)
                    logits = self.chance_projector(x)
                    probs = logits.softmax(dim=-1)
                    
                    one_hot = torch.zeros_like(probs).scatter_(
                        -1, torch.argmax(probs, dim=-1, keepdim=True), 1.0
                    )
                    # Straight-Through Estimator
                    codes_k = (one_hot - probs).detach() + probs

                stochastic_sequences["latents_afterstates"].append(afterstate)
                stochastic_sequences["chance_logits"].append(chance_logits)
                stochastic_sequences["afterstate_backbone_features"].append(shared_features)
                
                next_latent = self.dynamics(afterstate, codes_k)
            else:
                next_latent = self.dynamics(current_latent, action_k)

            # Heads Phase
            for name, head in self.heads.items():
                h_state = current_head_state.get(name) if isinstance(current_head_state, dict) else None
                logits, n_state, _ = head(next_latent, state=h_state)
                head_sequences[name].append(logits)
                if isinstance(current_head_state, dict):
                    current_head_state[name] = n_state

            current_latent = next_latent
            latents.append(current_latent)
            current_latent = scale_gradient(current_latent, 0.5)

        # Consistency Loss Phase (Optional)
        target_latents = None
        if "target_observations" in kwargs and self.config.consistency_loss_factor > 0:
            target_obs = kwargs["target_observations"]
            B_t, T_t = target_obs.shape[:2]
            flat_obs = target_obs.reshape(B_t * T_t, *target_obs.shape[2:])
            with torch.no_grad():
                encoded = self.representation(flat_obs.float())
                target_latents = encoded.view(B_t, T_t, *encoded.shape[1:])

        output = {"latents": torch.stack(latents, dim=1)}
        for name, seq in head_sequences.items():
            if seq: output[name] = torch.stack(seq, dim=1)
        for name, seq in stochastic_sequences.items():
            if seq: output[name] = torch.stack(seq, dim=1)

        if target_latents is not None:
            output["target_latents"] = target_latents

        return output

    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            "representation_network": self.representation,
            "dynamics_network": self.dynamics,
            "afterstate_dynamics_network": getattr(self, "afterstate_dynamics", None),
        }
