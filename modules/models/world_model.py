from typing import List, Optional, Tuple, Dict, Any, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from configs.agents.muzero import MuZeroConfig
from modules.backbones.factory import BackboneFactory
from modules.embeddings.action_fusion import ActionFusion
from modules.heads.factory import HeadFactory
from modules.utils import scale_gradient, _normalize_hidden_state
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
        latent_dimensions: Tuple[int, ...],
        num_actions: int,
    ):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_chance = config.num_chance
        
        # 1. Dynamics Networks (Action-conditioned backbones)
        if self.config.stochastic:
            # Afterstate Dynamics: (Latent, Action) -> Afterstate
            self.afterstate_fusion = ActionFusion(
                self.num_actions, self.config.action_embedding_dim, latent_dimensions
            )
            self.afterstate_dynamics = BackboneFactory.create(
                config.afterstate_dynamics_backbone, latent_dimensions
            )

            # Dynamics: (Afterstate, Chance) -> Next Latent
            self.dynamics_fusion = ActionFusion(
                self.num_chance, self.config.action_embedding_dim, latent_dimensions
            )
            self.dynamics = BackboneFactory.create(
                config.dynamics_backbone, latent_dimensions
            )

            self.sigma_head = HeadFactory.create(
                self.config.chance_probability_head,
                self.config.arch,
                input_shape=latent_dimensions,
                num_chance_codes=self.num_chance,
            )

            # Chance Encoder: (Stacked Obs) -> Chance Codes
            # Note: The Chance Encoder works directly on raw observations, normally this
            # belongs logically in the AgentNetwork's feature extraction, but it's specific
            # to the stochastic world model's chance generation. For now we leave it here,
            # but it uses the original observation_dimensions (which are now hidden from init).
            # We'll extract image sizes dynamically or assume config.observation_dimensions.
            # *WAIT, the config structure usually contains config.game.observation_shape.
            # Since we no longer receive it in __init__, we use the config's shape.
            encoder_input_shape = list(config.game.observation_shape)
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
            self.dynamics_fusion = ActionFusion(
                self.num_actions, self.config.action_embedding_dim, latent_dimensions
            )
            self.dynamics = BackboneFactory.create(
                config.dynamics_backbone, latent_dimensions
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



    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        recurrent_state: Any = None,
    ) -> WorldModelOutput:
        # 1. Action Fusion Phase
        fused_latent = self.dynamics_fusion(hidden_state, action)
        
        # 2. Backbone Transition Phase
        next_hidden_state = self.dynamics(fused_latent)
        
        # 3. MuZero Hidden State Normalization
        next_hidden_state = _normalize_hidden_state(next_hidden_state)
        
        predictions = {}
        head_state = {} if recurrent_state is None else recurrent_state
        new_head_state = {}

        for name, head in self.heads.items():
            h_state = head_state.get(name) if isinstance(head_state, dict) else None
            head_out = head(next_hidden_state, state=h_state)
            predictions[name] = head_out.training_tensor
            predictions[f"{name}_extra"] = head_out.inference_tensor
            new_head_state[name] = head_out.state

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

    def _step_dynamics(
        self,
        current_latent: Tensor,
        action_k: Tensor,
        k: int,
        encoder_inputs: Optional[Tensor] = None,
        true_chance_codes: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Processes a single transition step (Deterministic or Stochastic)."""
        if self.config.stochastic:
            afterstate_out = self.afterstate_recurrent_inference(
                {"dynamics": current_latent}, action_k
            )
            afterstate = afterstate_out.afterstate_features
            chance_logits = afterstate_out.chance

            # 2. Chance Code Selection (Ground Truth or Learned Encoder)
            if self.config.use_true_chance_codes and true_chance_codes is not None:
                codes_k = F.one_hot(
                    true_chance_codes[:, k + 1].squeeze(-1).long(),
                    self.num_chance,
                ).float()
            else:
                # Straight-Through Estimator logic for backpropping to encoder
                # Note: k index for encoder matches root=0, k=step
                x = self.encoder(encoder_inputs[:, k])
                x = x.flatten(1, -1)
                logits = self.chance_projector(x)
                probs = logits.softmax(dim=-1)
                one_hot = torch.zeros_like(probs).scatter_(
                    -1, torch.argmax(probs, dim=-1, keepdim=True), 1.0
                )
                codes_k = (one_hot - probs).detach() + probs

            # 3. Transition Phase (Complete hidden state update)
            next_latent = self.dynamics_fusion(afterstate, codes_k)
            next_latent = self.dynamics(next_latent)
            next_latent = _normalize_hidden_state(next_latent)

            return {
                "next_latent": next_latent,
                "afterstate": afterstate,
                "chance_logits": chance_logits,
            }
        else:
            # Standard Deterministic Dynamics
            next_latent = self.dynamics_fusion(current_latent, action_k)
            next_latent = self.dynamics(next_latent)
            next_latent = _normalize_hidden_state(next_latent)
            return {"next_latent": next_latent}

    def afterstate_recurrent_inference(
        self,
        network_state: Dict[str, Any],
        action: Tensor,
    ) -> WorldModelOutput:
        latent_state = network_state["dynamics"]
        
        fused_latent = self.afterstate_fusion(latent_state, action)
        afterstate_latent = self.afterstate_dynamics(fused_latent)
        afterstate_latent = _normalize_hidden_state(afterstate_latent)
        
        head_out_sigma = self.sigma_head(afterstate_latent)
        chance_logits = head_out_sigma.training_tensor

        return WorldModelOutput(
            features=torch.empty(0), # Ignored in afterstate path
            afterstate_features=afterstate_latent,
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
            head_out = head(current_latent, state=h_state)
            head_sequences[name].append(head_out.training_tensor)
            if isinstance(current_head_state, dict):
                current_head_state[name] = head_out.state

        stochastic_sequences = {
            "latents_afterstates": [],
            "chance_logits": [],
        } if self.config.stochastic else {}

        for k in range(unroll_steps):
            action_k = actions[:, k]

            # 1. Step Dynamics Phase (Deterministic vs Stochastic)
            step_results = self._step_dynamics(
                current_latent,
                action_k,
                k,
                encoder_inputs=encoder_inputs,
                true_chance_codes=true_chance_codes,
            )
            next_latent = step_results["next_latent"]

            # 2. Optional Metadata Recording (Stochastic paths only)
            if self.config.stochastic:
                stochastic_sequences["latents_afterstates"].append(
                    step_results["afterstate"]
                )
                stochastic_sequences["chance_logits"].append(
                    step_results["chance_logits"]
                )

            # Heads Phase
            for name, head in self.heads.items():
                h_state = current_head_state.get(name) if isinstance(current_head_state, dict) else None
                head_out = head(next_latent, state=h_state)
                head_sequences[name].append(head_out.training_tensor)
                if isinstance(current_head_state, dict):
                    current_head_state[name] = head_out.state

            current_latent = next_latent
            latents.append(current_latent)
            current_latent = scale_gradient(current_latent, 0.5)



        output = {"latents": torch.stack(latents, dim=1)}
        for name, seq in head_sequences.items():
            if seq: output[name] = torch.stack(seq, dim=1)
        for name, seq in stochastic_sequences.items():
            if seq: output[name] = torch.stack(seq, dim=1)

        return output

    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            "dynamics_fusion": self.dynamics_fusion,
            "dynamics_backbone": self.dynamics,
            "afterstate_fusion": getattr(self, "afterstate_fusion", None),
            "afterstate_backbone": getattr(self, "afterstate_dynamics", None),
        }
