from typing import List, Optional, Tuple, Dict, Any, Union, Callable
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from modules.embeddings.action_fusion import ActionFusion
from modules.utils import scale_gradient, _normalize_hidden_state
from modules.models.inference_output import WorldModelOutput


class DeterministicDynamics(nn.Module):
    def __init__(
        self,
        latent_dimensions: Tuple[int, ...],
        num_actions: int,
        dynamics_fn: Callable[[Tuple[int, ...]], nn.Module],
        action_embedding_dim: int,
    ):
        super().__init__()
        self.dynamics_fusion = ActionFusion(
            num_actions, action_embedding_dim, latent_dimensions
        )
        self.dynamics = dynamics_fn(input_shape=latent_dimensions)
        self.output_shape = self.dynamics.output_shape

    def forward(
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
        num_chance: int,
        observation_shape: Tuple[int, ...],
        dynamics_fn: Callable[[Tuple[int, ...]], nn.Module],
        afterstate_dynamics_fn: Callable[[Tuple[int, ...]], nn.Module],
        sigma_head_fn: Callable[..., nn.Module],
        encoder_fn: Callable[[Tuple[int, ...]], nn.Module],
        action_embedding_dim: int,
        use_true_chance_codes: bool = False,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_chance = num_chance
        self.use_true_chance_codes = use_true_chance_codes
        self.prediction_backbone = nn.Identity()

        # 1. Afterstate Phase
        self.afterstate_fusion = ActionFusion(
            num_actions, action_embedding_dim, latent_dimensions
        )
        self.afterstate_dynamics = afterstate_dynamics_fn(input_shape=latent_dimensions)

        # 2. Dynamics Phase
        self.dynamics_fusion = ActionFusion(
            self.num_chance, action_embedding_dim, latent_dimensions
        )
        self.dynamics = dynamics_fn(input_shape=latent_dimensions)
        self.output_shape = self.dynamics.output_shape

        # 3. Chance Prediction
        self.sigma_head = sigma_head_fn(
            input_shape=latent_dimensions,
            num_chance_codes=self.num_chance,
        )

        # 4. Chance Encoder
        encoder_input_shape = list(observation_shape)
        encoder_input_shape[0] *= 2  # Double channels for stacked obs
        self.encoder = encoder_fn(input_shape=tuple(encoder_input_shape))
        # Map flattened backbone output to codes using a foolproof dummy pass
        from modules.utils import get_flat_dim
        flat_dim = get_flat_dim(self.encoder, tuple(encoder_input_shape))
        self.chance_projector = nn.Linear(flat_dim, self.num_chance)

    def forward(
        self,
        current_latent: Tensor,
        action: Tensor,
        **kwargs,
    ) -> Dict[str, Tensor]:
        # Full transition for Learner unrolls
        k = kwargs.get("k", 0)
        encoder_inputs = kwargs.get("encoder_inputs", None)
        true_chance_codes = kwargs.get("true_chance_codes", None)
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
            "afterstate_features": afterstate,
            "chance_logits": afterstate_res["chance_logits"],
            "chance_dist": afterstate_res["chance_dist"],
        }

    def afterstate_inference(self, latent: Tensor, action: Tensor) -> Dict[str, Any]:
        fused = self.afterstate_fusion(latent, action)
        afterstate = self.afterstate_dynamics(fused)
        afterstate = _normalize_hidden_state(afterstate)

        # Apply the shared prediction backbone (MuZero parity)
        # Fallback to identity if not explicitly set via set_prediction_backbone
        processed_afterstate = self.prediction_backbone(afterstate)
        head_out = self.sigma_head(processed_afterstate, is_inference=True)
        return {
            "afterstate_features": afterstate,
            "processed_afterstate": processed_afterstate,
            "chance_logits": head_out.training_tensor,
            "chance_dist": head_out.inference_tensor,
        }

    def set_prediction_backbone(self, backbone: nn.Module):
        self.prediction_backbone = backbone

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
        latent_dimensions: Tuple[int, ...],
        num_actions: int,
        stochastic: bool = False,
        num_chance: int = 0,
        observation_shape: Optional[Tuple[int, ...]] = None,
        use_true_chance_codes: bool = False,
        num_players: int = 1,
        env_head_fns: Dict[str, Callable[..., nn.Module]] = None,
        dynamics_fn: Callable[[Tuple[int, ...]], nn.Module] = None,
        afterstate_dynamics_fn: Callable[[Tuple[int, ...]], nn.Module] = None,
        sigma_head_fn: Callable[..., nn.Module] = None,
        encoder_fn: Callable[[Tuple[int, ...]], nn.Module] = None,
        action_embedding_dim: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.stochastic = stochastic

        # 1. Dynamics Strategy
        if self.stochastic:
            self.dynamics_pipeline = StochasticDynamics(
                latent_dimensions=latent_dimensions,
                num_actions=num_actions,
                num_chance=num_chance,
                observation_shape=observation_shape,
                dynamics_fn=dynamics_fn,
                afterstate_dynamics_fn=afterstate_dynamics_fn,
                sigma_head_fn=sigma_head_fn,
                encoder_fn=encoder_fn,
                action_embedding_dim=action_embedding_dim,
                use_true_chance_codes=use_true_chance_codes,
            )
        else:
            self.dynamics_pipeline = DeterministicDynamics(
                latent_dimensions=latent_dimensions,
                num_actions=num_actions,
                dynamics_fn=dynamics_fn,
                action_embedding_dim=action_embedding_dim,
            )

        # 2. Environment Heads
        self.heads = nn.ModuleDict()

        # Iterate over configured environment heads
        if env_head_fns:
            for head_name, head_fn in env_head_fns.items():
                if head_fn is None:
                    continue

                self.heads[head_name] = head_fn(
                    input_shape=self.dynamics_pipeline.output_shape,
                    num_players=num_players,
                    num_actions=num_actions,
                    num_chance_codes=num_chance,
                    name=head_name,
                )

        self.register_buffer("_device_indicator", torch.empty(0))

    def set_prediction_backbone(self, backbone: nn.Module):
        if hasattr(self, "dynamics_pipeline") and hasattr(self.dynamics_pipeline, "set_prediction_backbone"):
            self.dynamics_pipeline.set_prediction_backbone(backbone)

    @property
    def device(self) -> torch.device:
        """Determines the device the world model is currently on using a buffer indicator."""
        return self._device_indicator.device



    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        recurrent_state: Any = None,
        **kwargs,
    ) -> WorldModelOutput:
        # Both implement the forward() contract.
        step_results = self.dynamics_pipeline(hidden_state, action, **kwargs)
        next_hidden_state = step_results["next_latent"]

        # 2. Heads Phase
        predictions = {}
        head_state = {} if recurrent_state is None else recurrent_state
        new_head_state = {}

        for name, head in self.heads.items():
            head_out = head(
                next_hidden_state, state=head_state, is_inference=True, **kwargs
            )
            predictions[name] = head_out.training_tensor
            predictions[f"{name}_extra"] = head_out.inference_tensor

            if head_out.state:
                new_head_state.update(head_out.state)

        # The World Model packs everything it owns into one dictionary
        new_state = {"dynamics": next_hidden_state, **new_head_state}

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
            features=res["processed_afterstate"],
            afterstate_features=res["afterstate_features"],
            chance=res["chance_logits"],
            chance_dist=res["chance_dist"],
            next_state=new_state,
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
            head_out = head(
                current_latent, state=current_head_state, is_inference=False, **kwargs
            )
            head_sequences[name].append(head_out.training_tensor)

            current_head_state.update(head_out.state)

        stochastic_sequences = (
            {"latents_afterstates": [], "chance_logits": []} if self.stochastic else {}
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
            if "afterstate_features" in step_results:
                stochastic_sequences["latents_afterstates"].append(
                    step_results["afterstate_features"]
                )
            if "chance_logits" in step_results:
                stochastic_sequences["chance_logits"].append(
                    step_results["chance_logits"]
                )

            # 3. Heads Phase
            for name, head in self.heads.items():
                head_out = head(
                    next_latent, state=current_head_state, is_inference=False, **kwargs
                )
                head_sequences[name].append(head_out.training_tensor)

                current_head_state.update(head_out.state)

            current_latent = next_latent
            latents.append(current_latent)
            current_latent = scale_gradient(current_latent, 0.5)

        output = {"latents": torch.stack(latents, dim=1)}
        # Merge dictionaries and stack uniformly
        for name, seq in {**head_sequences, **stochastic_sequences}.items():
            if seq:
                output[name] = torch.stack(seq, dim=1)

        return output
