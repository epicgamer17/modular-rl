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


class DeterministicDynamics(nn.Module):
    def __init__(
        self,
        latent_dimensions: Tuple[int, ...],
        num_actions: int,
        world_model_config: Any,
    ):
        super().__init__()
        self.dynamics_fusion = ActionFusion(
            num_actions, world_model_config.action_embedding_dim, latent_dimensions
        )
        self.dynamics = BackboneFactory.create(
            world_model_config.dynamics_backbone, latent_dimensions
        )
        self.output_shape = self.dynamics.output_shape

    def recurrent_step(
        self, current_latent: Tensor, action: Tensor, **kwargs
    ) -> Dict[str, Tensor]:
        next_latent = self.dynamics_fusion(current_latent, action)
        next_latent = self.dynamics(next_latent)
        next_latent = _normalize_hidden_state(next_latent)
        return {"next_latent": next_latent}


class StochasticDynamics(nn.Module):
    def __init__(
        self,
        latent_dimensions: Tuple[int, ...],
        num_actions: int,
        world_model_config: Any,
        num_chance: int,
        arch_config: Any,
        observation_shape: Tuple[int, ...],
        use_true_chance_codes: bool = False,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_chance = num_chance
        self.use_true_chance_codes = use_true_chance_codes

        # 1. Afterstate Phase
        self.afterstate_fusion = ActionFusion(
            num_actions, world_model_config.action_embedding_dim, latent_dimensions
        )
        self.afterstate_dynamics = BackboneFactory.create(
            world_model_config.afterstate_dynamics_backbone, latent_dimensions
        )

        # 2. Dynamics Phase
        self.dynamics_fusion = ActionFusion(
            self.num_chance, world_model_config.action_embedding_dim, latent_dimensions
        )
        self.dynamics = BackboneFactory.create(
            world_model_config.dynamics_backbone, latent_dimensions
        )
        self.output_shape = self.dynamics.output_shape

        # 3. Chance Prediction
        self.sigma_head = HeadFactory.create(
            world_model_config.chance_probability_head,
            arch_config,
            input_shape=latent_dimensions,
            num_chance_codes=self.num_chance,
        )

        # 4. Chance Encoder
        encoder_input_shape = list(observation_shape)
        encoder_input_shape[0] *= 2  # Double channels for stacked obs
        self.encoder = BackboneFactory.create(
            world_model_config.chance_encoder_backbone, tuple(encoder_input_shape)
        )
        # Map flattened backbone output to codes
        flat_dim = 1
        for d in self.encoder.output_shape:
            flat_dim *= d
        self.chance_projector = nn.Linear(flat_dim, self.num_chance)

    def forward(
        self,
        current_latent: Tensor,
        action: Tensor,
        k: int,
        encoder_inputs: Optional[Tensor] = None,
        true_chance_codes: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        # Full transition for Learner unrolls
        afterstate_res = self.afterstate_inference(current_latent, action)
        afterstate = afterstate_res["afterstate_features"]

        # Chance Selection
        if self.use_true_chance_codes and true_chance_codes is not None:
            codes_k = F.one_hot(
                true_chance_codes[:, k + 1].squeeze(-1).long(),
                self.num_chance,
            ).float()
        else:
            # Straight-Through Estimator logic
            x = self.encoder(encoder_inputs[:, k])
            x = x.flatten(1, -1)
            logits = self.chance_projector(x)
            probs = logits.softmax(dim=-1)
            one_hot = torch.zeros_like(probs).scatter_(
                -1, torch.argmax(probs, dim=-1, keepdim=True), 1.0
            )
            codes_k = (one_hot - probs).detach() + probs

        next_latent = self.recurrent_step(afterstate, codes_k)["next_latent"]

        return {
            "next_latent": next_latent,
            "afterstate": afterstate,
            "chance_logits": afterstate_res["chance"],
        }

    def afterstate_inference(self, latent: Tensor, action: Tensor) -> Dict[str, Tensor]:
        fused = self.afterstate_fusion(latent, action)
        afterstate = self.afterstate_dynamics(fused)
        afterstate = _normalize_hidden_state(afterstate)
        chance_logits = self.sigma_head(afterstate).training_tensor
        return {"afterstate_features": afterstate, "chance": chance_logits}

    def recurrent_step(self, state: Tensor, chance_code: Tensor) -> Dict[str, Tensor]:
        fused = self.dynamics_fusion(state, chance_code)
        next_latent = self.dynamics(fused)
        return {"next_latent": _normalize_hidden_state(next_latent)}


class WorldModel(nn.Module):
    """
    A modular world model that encapsulates the representation, dynamics,
    and environment heads.
    """

    def __init__(
        self,
        config: Any,
        latent_dimensions: Tuple[int, ...],
        num_actions: int,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.stochastic = getattr(config, "stochastic", False)

        # 1. Dynamics Strategy
        if self.stochastic:
            self.dynamics_pipeline = StochasticDynamics(
                latent_dimensions=latent_dimensions,
                num_actions=num_actions,
                world_model_config=config,
                num_chance=getattr(config, "num_chance", 0),
                arch_config=config.arch,
                observation_shape=config.game.observation_shape,
                use_true_chance_codes=getattr(config, "use_true_chance_codes", False),
            )
        else:
            self.dynamics_pipeline = DeterministicDynamics(
                latent_dimensions=latent_dimensions,
                num_actions=num_actions,
                world_model_config=config,
            )

        # 2. Environment Heads
        self.heads = nn.ModuleDict()

        # Iterate over configured environment heads
        if hasattr(config, "env_heads"):
            for head_name, head_config in config.env_heads.items():
                if head_config is None:
                    continue

                self.heads[head_name] = HeadFactory.create(
                    head_config,
                    arch_config=config.arch,
                    input_shape=self.dynamics_pipeline.output_shape,
                    num_players=config.game.num_players,
                    num_actions=num_actions,
                    num_chance_codes=getattr(config, "num_chance", 0),
                )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def sigma_head(self) -> Optional[nn.Module]:
        if not self.stochastic:
            return None
        return self.dynamics_pipeline.sigma_head

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        recurrent_state: Any = None,
        **kwargs,
    ) -> WorldModelOutput:
        # 1. Transition Phase (Via Pipeline)
        # The World Model no longer cares if it's stochastic or deterministic!
        # Both implement the .recurrent_step() contract.
        next_hidden_state = self.dynamics_pipeline.recurrent_step(hidden_state, action)[
            "next_latent"
        ]

        # 2. Heads Phase
        predictions = {}
        head_state = {} if recurrent_state is None else recurrent_state
        new_head_state = {}

        for name, head in self.heads.items():
            head_out = head(next_hidden_state, state=head_state, **kwargs)
            predictions[name] = head_out.training_tensor
            predictions[f"{name}_extra"] = head_out.inference_tensor

            if head_out.state:
                new_head_state.update(head_out.state)

        # The World Model packs everything it owns into one dictionary
        new_state = {
            "dynamics": next_hidden_state,
            **new_head_state
        }

        return WorldModelOutput(
            features=next_hidden_state,
            reward=predictions.get("reward_logits"),
            to_play_logits=predictions.get("to_play_logits"),
            to_play=predictions.get("to_play_logits_extra"),
            continuation_logits=predictions.get("continuation_logits"),
            continuation=predictions.get("continuation_logits_extra"),
            next_state=new_state,
            instant_reward=predictions.get("reward_logits_extra"),
        )

    def afterstate_recurrent_inference(
        self,
        network_state: Dict[str, Any],
        action: Tensor,
    ) -> WorldModelOutput:
        if not self.stochastic:
            raise NotImplementedError(
                "afterstate_recurrent_inference requires a stochastic world_model."
            )

        res = self.dynamics_pipeline.afterstate_inference(
            network_state["dynamics"], action
        )

        new_state = {
            "dynamics": res["afterstate_features"],
        }

        return WorldModelOutput(
            features=torch.empty(0),
            afterstate_features=res["afterstate_features"],
            chance=res["chance"],
            next_state=new_state
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
        current_head_state: Dict[str, Any] = (
            head_state if head_state is not None else {}
        )

        # Initial head prediction for root
        for name, head in self.heads.items():
            head_out = head(current_latent, state=current_head_state, **kwargs)
            head_sequences[name].append(head_out.training_tensor)

            if head_out.state:
                current_head_state.update(head_out.state)

        stochastic_sequences = (
            {"latents_afterstates": [], "chance_logits": []}
            if self.stochastic
            else {}
        )

        for k in range(unroll_steps):
            action_k = actions[:, k]

            # 1. Step Dynamics Phase (Generic Pipeline)
            step_results = self.dynamics_pipeline(
                current_latent,
                action_k,
                k=k,
                encoder_inputs=encoder_inputs,
                true_chance_codes=true_chance_codes,
            )
            next_latent = step_results["next_latent"]

            # 2. Metadata Recording
            if "afterstate" in step_results:
                stochastic_sequences["latents_afterstates"].append(
                    step_results["afterstate"]
                )
            if "chance_logits" in step_results:
                stochastic_sequences["chance_logits"].append(
                    step_results["chance_logits"]
                )

            # 3. Heads Phase
            for name, head in self.heads.items():
                head_out = head(next_latent, state=current_head_state, **kwargs)
                head_sequences[name].append(head_out.training_tensor)

                if head_out.state:
                    current_head_state.update(head_out.state)

            current_latent = next_latent
            latents.append(current_latent)
            current_latent = scale_gradient(current_latent, 0.5)

        output = {"latents": torch.stack(latents, dim=1)}
        for name, seq in head_sequences.items():
            if seq:
                output[name] = torch.stack(seq, dim=1)
        for name, seq in stochastic_sequences.items():
            if seq:
                output[name] = torch.stack(seq, dim=1)

        return output
