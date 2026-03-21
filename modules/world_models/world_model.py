from typing import Callable, List, Optional, Tuple, Dict, Any, Union

from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F

from configs.agents.muzero import MuZeroConfig
from modules.backbones.factory import BackboneFactory
from modules.heads.factory import HeadFactory
from modules.utils import scale_gradient
from modules.world_models.inference_output import WorldModelOutput

from modules.world_models.components.representation import Representation
from modules.world_models.components.dynamics import Dynamics, AfterstateDynamics
from modules.world_models.components.chance_encoder import ChanceEncoder

class WorldModel(nn.Module):
    """
    A modular world model that encapsulates the representation, dynamics, 
    and environment heads (Reward, Continuation, To-Play).
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
        
        # 1. Representation Network
        self.representation = Representation(config, observation_dimensions)
        self.num_chance = config.num_chance

        # 2. Dynamics Networks
        if self.config.stochastic:
            self.afterstate_dynamics = AfterstateDynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
            )

            self.dynamics = Dynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_chance,
                action_embedding_dim=self.config.action_embedding_dim,
            )

            self.shared_backbone = BackboneFactory.create(
                self.config.prediction_backbone, self.representation.output_shape
            )

            self.sigma_head = HeadFactory.create(
                self.config.chance_probability_head,
                self.config.arch,
                input_shape=self.shared_backbone.output_shape,
                num_chance_codes=self.num_chance,
            )

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

        assert (
            self.dynamics.output_shape == self.representation.output_shape
        ), f"{self.dynamics.output_shape} != {self.representation.output_shape}"

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
        if not self.config.stochastic:
            action = action.view(-1).to(hidden_state.device)
            action = F.one_hot(action.long(), num_classes=self.num_actions).float().to(hidden_state.device)

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
            to_play=predictions.get("to_play_logits_extra"),
            to_play_logits=predictions.get("to_play_logits"),
            continuation=predictions.get("continuation_logits_extra"),
            continuation_logits=predictions.get("continuation_logits"),
            head_state=new_head_state,
            instant_reward=predictions.get("reward_logits_extra"),
        )

    def afterstate_recurrent_inference(
        self,
        network_state: Dict[str, Any],
        action: Tensor,
    ) -> WorldModelOutput:
        latent_state = network_state["dynamics"]
        action = action.view(-1).to(latent_state.device)
        action = F.one_hot(action.long(), num_classes=self.num_actions).float().to(latent_state.device)

        afterstate_latent = self.afterstate_dynamics(latent_state, action)
        shared_features = self.shared_backbone(afterstate_latent)
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
        batch_size, unroll_steps = actions.shape[:2]
        latents = [initial_latent_state]
        head_sequences = {name: [] for name in self.heads.keys()}
        
        current_latent = initial_latent_state
        current_head_state = head_state if head_state is not None else {}
        
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
                    codes_k = F.one_hot(
                        true_chance_codes[:, k + 1].squeeze(-1).long(),
                        self.config.num_chance,
                    ).float()
                else:
                    _, codes_k = self.encoder(encoder_inputs[:, k])

                stochastic_sequences["latents_afterstates"].append(afterstate)
                stochastic_sequences["chance_logits"].append(chance_logits)
                stochastic_sequences["afterstate_backbone_features"].append(shared_features)
                
                next_latent = self.dynamics(afterstate, codes_k)
            else:
                action_k_vec = action_k.view(-1).long()
                action_onehot = F.one_hot(action_k_vec, num_classes=self.num_actions).float()
                next_latent = self.dynamics(current_latent, action_onehot)

            for name, head in self.heads.items():
                h_state = current_head_state.get(name) if isinstance(current_head_state, dict) else None
                logits, n_state, _ = head(next_latent, state=h_state)
                head_sequences[name].append(logits)
                if isinstance(current_head_state, dict):
                    current_head_state[name] = n_state

            current_latent = next_latent
            latents.append(current_latent)
            current_latent = scale_gradient(current_latent, 0.5)

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
