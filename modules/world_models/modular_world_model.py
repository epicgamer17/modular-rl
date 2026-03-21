from typing import Callable, List, Optional, Tuple, Dict, Any, Union

from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F

from configs.agents.muzero import MuZeroConfig
from modules.backbones.factory import BackboneFactory
from modules.heads.factory import HeadFactory
from modules.heads.to_play import ToPlayHead
from modules.heads.reward import RewardHead
from agents.learner.losses.representations import get_representation
from modules.utils import scale_gradient, kernel_initializer_wrapper
from modules.world_models.inference_output import (
    WorldModelOutput,
)
from modules.world_models.world_model import WorldModelInterface

from modules.world_models.components.representation import Representation
from modules.world_models.components.dynamics import Dynamics, AfterstateDynamics
from modules.world_models.components.chance_encoder import ChanceEncoder

from modules.world_models.world_model import WorldModelOutput


class ModularWorldModel(WorldModelInterface, nn.Module):
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

        # 3. Environment Heads (Physics Engine)
        self.heads = nn.ModuleDict()
        
        # Reward Head
        if getattr(config, "reward_head", None) is not None:
            r_rep = get_representation(config.reward_head.output_strategy)
            self.heads["reward_logits"] = HeadFactory.create(
                config.reward_head,
                config.arch,
                input_shape=self.dynamics.output_shape,
                representation=r_rep,
            )

        # Continuation Head (Termination)
        if getattr(config, "continuation_head", None) is not None:
            c_rep = get_representation(config.continuation_head.output_strategy)
            self.heads["continuation_logits"] = HeadFactory.create(
                config.continuation_head,
                config.arch,
                input_shape=self.dynamics.output_shape,
                representation=c_rep,
            )

        # To-Play Head
        if getattr(config, "to_play_head", None) is not None:
            tp_rep = get_representation(config.to_play_head.output_strategy)
            self.heads["to_play_logits"] = HeadFactory.create(
                config.to_play_head,
                config.arch,
                input_shape=self.dynamics.output_shape,
                num_players=config.game.num_players,
                representation=tp_rep,
            )

        # Dynamics output must match Representation output shape
        assert (
            self.dynamics.output_shape == self.representation.output_shape
        ), f"{self.dynamics.output_shape} = {self.representation.output_shape}"

    @property
    def device(self) -> torch.device:
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )


    def initial_inference(self, observation: Tensor) -> WorldModelOutput:
        # Ensure observation is a tensor
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            )

        # Ensure observation has batch dim
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
            # one-hot the action -> (B, num_actions)
            action = (
                F.one_hot(action.long(), num_classes=self.num_actions)
                .float()
                .to(hidden_state.device)
            )

        next_hidden_state = self.dynamics(hidden_state, action)
        
        # Predictions for all heads
        predictions = {}
        head_state = {} if recurrent_state is None else recurrent_state
        new_head_state = {}

        for name, head in self.heads.items():
            # Pass individual head state if present
            h_state = head_state.get(name) if isinstance(head_state, dict) else None
            
            # Heads return (logits, state, extra)
            # extra is usually the expected value (reward, player_idx)
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
        """
        Computes the afterstate from a deterministic action (stochastic MuZero).

        This method is called from two places:
        - ``unroll_physics`` (internal): passes a raw latent Tensor directly.
        - ``ModularAgentNetwork.afterstate_inference`` (MCTS external): passes the
          opaque ``{"dynamics": Tensor, "wm_memory": ...}`` dict.

        Both cases are handled gracefully.

        Args:
            network_state: A dictionary with 'dynamics' and optional 'wm_memory' —
                either from ``ModularAgentNetwork.afterstate_inference`` (MCTS) or
                wrapped by ``unroll_physics`` (learner). Must never be a raw Tensor.
            action: Action taken at this step.

        Returns:
            WorldModelOutput with afterstate features and chance logits.
        """
        latent_state = network_state["dynamics"]

        # 1. Transition to Afterstate
        action = action.view(-1).to(latent_state.device)
        # one-hot the action -> (B, num_actions)
        action = (
            F.one_hot(action.long(), num_classes=self.num_actions)
            .float()
            .to(latent_state.device)
        )

        afterstate_latent = self.afterstate_dynamics(latent_state, action)

        # 2. Extract Shared Features (Backbone)
        shared_features = self.shared_backbone(afterstate_latent)

        # 3. Predict Chance Probabilities (Logits)
        chance_logits, _, _ = self.sigma_head(shared_features)

        return WorldModelOutput(
            afterstate_features=afterstate_latent,  # Raw dynamics output
            features=shared_features,  # Shared processed features for Agent
            chance=chance_logits,  # Environment chance rules
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
        """
        Unrolls the dynamics for K steps given actions.
        Returns a dictionary containing sequences for each predicted field.
        """
        batch_size, unroll_steps = actions.shape[:2]
        device = initial_latent_state.device

        # 1. Initialize sequences with root state
        latents = [initial_latent_state]
        head_sequences = {name: [] for name in self.heads.keys()}
        
        current_latent = initial_latent_state
        current_head_state = head_state if head_state is not None else {}
        
        # Predict all heads at root
        for name, head in self.heads.items():
            h_state = current_head_state.get(name) if isinstance(current_head_state, dict) else None
            # At root, reward is typically 0, but to_play is essential
            logits, n_state, _ = head(current_latent, state=h_state)
            head_sequences[name].append(logits)
            if isinstance(current_head_state, dict):
                current_head_state[name] = n_state

        # Stochastic-specific storage
        stochastic_sequences = {
            "latents_afterstates": [],
            "chance_logits": [],
            "afterstate_backbone_features": [],
            "chance_encoder_embeddings": [],
            "chance_encoder_onehots": [],
        } if self.config.stochastic else {}

        # 2. Step Dynamics and Heads (Steps 1...K)
        for k in range(unroll_steps):
            action_k = actions[:, k]

            if self.config.stochastic:
                # Afterstate Recurrent Inference
                afterstate_out = self.afterstate_recurrent_inference(
                    {"dynamics": current_latent}, action_k
                )
                afterstate = afterstate_out.afterstate_features
                shared_features = afterstate_out.features
                chance_logits = afterstate_out.chance

                # Chance Code
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
                
                # Final Dynamics Step
                next_latent = self.dynamics(afterstate, codes_k)
            else:
                # Deterministic Step
                action_k_vec = action_k.view(-1).long()
                action_onehot = F.one_hot(action_k_vec, num_classes=self.num_actions).float()
                next_latent = self.dynamics(current_latent, action_onehot)

            # 3. Predict Environment Heads
            # We step ALL heads defined in self.heads
            for name, head in self.heads.items():
                h_state = current_head_state.get(name) if isinstance(current_head_state, dict) else None
                logits, n_state, _ = head(next_latent, state=h_state)
                head_sequences[name].append(logits)
                # Update head state for next step
                if isinstance(current_head_state, dict):
                    current_head_state[name] = n_state

            current_latent = next_latent
            latents.append(current_latent)
            
            # Scale Gradient
            current_latent = scale_gradient(current_latent, 0.5)

        # 4. Target Latents (Consistency Loss)
        target_latents = None
        if "target_observations" in kwargs and self.config.consistency_loss_factor > 0:
            target_obs = kwargs["target_observations"]
            B_t, T_t = target_obs.shape[:2]
            flat_obs = target_obs.reshape(B_t * T_t, *target_obs.shape[2:])
            with torch.no_grad():
                encoded = self.representation(flat_obs.float())
                target_latents = encoded.view(B_t, T_t, *encoded.shape[1:])

        # 5. Stack Final Output
        # Dynamics sequences: [B, T+1, ...]
        output = {
            "latents": torch.stack(latents, dim=1),
        }
        
        # Head sequences: [B, T, ...]
        for name, seq in head_sequences.items():
            if seq:
                output[name] = torch.stack(seq, dim=1)

        # Stochastic sequences: [B, T, ...]
        for name, seq in stochastic_sequences.items():
            if seq:
                output[name] = torch.stack(seq, dim=1)

        if target_latents is not None:
            output["target_latents"] = target_latents

        return output

    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            "representation_network": self.representation,
            "dynamics_network": self.dynamics,
            "afterstate_dynamics_network": getattr(self, "afterstate_dynamics", None),
        }
