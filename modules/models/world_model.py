from typing import List, Optional, Tuple, Dict, Any, Union, Callable
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from modules.embeddings.action_fusion import ActionFusion
from modules.embeddings.action_embedding import ActionEncoder, get_action_encoder
from modules.utils import scale_gradient, _normalize_hidden_state
from modules.models.inference_output import WorldModelOutput


class DeterministicDynamics(nn.Module):
    def __init__(
        self,
        latent_dimensions: Tuple[int, ...],
        dynamics_fn: Callable[[Tuple[int, ...]], nn.Module],
        action_encoder: ActionEncoder,
        use_bn: bool = False,
    ):
        super().__init__()
        self.dynamics_fusion = ActionFusion(
            action_encoder, latent_dimensions, use_bn=use_bn
        )
        self.dynamics = dynamics_fn(input_shape=latent_dimensions)
        self.output_shape = getattr(self.dynamics, "output_shape", latent_dimensions)

    def forward(
        self, current_latent: Tensor, action: Tensor, **kwargs
    ) -> Dict[str, Tensor]:
        next_latent_unnorm = self.dynamics_fusion(current_latent, action)
        next_latent_unnorm = self.dynamics(next_latent_unnorm)

        next_latent_norm = _normalize_hidden_state(next_latent_unnorm)
        return {
            "next_latent": next_latent_norm,
            "unnormalized_latent": next_latent_unnorm,
        }


class StochasticDynamics(nn.Module):
    def __init__(
        self,
        latent_dimensions: Tuple[int, ...],
        num_chance: int,
        observation_shape: Tuple[int, ...],
        dynamics_fn: Callable[[Tuple[int, ...]], nn.Module],
        afterstate_dynamics_fn: Callable[[Tuple[int, ...]], nn.Module],
        sigma_head_fn: Callable[..., nn.Module],
        encoder_fn: Callable[[Tuple[int, ...]], nn.Module],
        action_encoder: ActionEncoder,
        chance_encoder: ActionEncoder,
        use_true_chance_codes: bool = False,
        use_bn: bool = False,
    ):
        super().__init__()
        self.num_chance = num_chance
        self.use_true_chance_codes = use_true_chance_codes

        # 1. Afterstate Phase
        self.afterstate_fusion = ActionFusion(
            action_encoder, latent_dimensions, use_bn=use_bn
        )
        self.afterstate_dynamics = afterstate_dynamics_fn(input_shape=latent_dimensions)

        # 2. Dynamics Phase
        self.dynamics_fusion = ActionFusion(
            chance_encoder, latent_dimensions, use_bn=use_bn
        )
        self.dynamics = dynamics_fn(input_shape=latent_dimensions)
        self.output_shape = getattr(self.dynamics, "output_shape", latent_dimensions)

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

        next_latent_res = self.recurrent_step(afterstate, codes_k)
        next_latent = next_latent_res["next_latent"]

        return {
            "next_latent": next_latent,
            "unnormalized_latent": next_latent_res["unnormalized_latent"],
            "afterstate_features": afterstate,
            "chance_logits": afterstate_res["chance_logits"],
            "chance_dist": afterstate_res["chance_dist"],
        }

    def afterstate_inference(self, latent: Tensor, action: Tensor) -> Dict[str, Any]:
        fused = self.afterstate_fusion(latent, action)
        afterstate = self.afterstate_dynamics(fused)
        afterstate = _normalize_hidden_state(afterstate)

        head_out = self.sigma_head(afterstate, is_inference=True)
        return {
            "afterstate_features": afterstate,
            "chance_logits": head_out.training_tensor,
            "chance_dist": head_out.inference_tensor,
        }

    def recurrent_step(self, state: Tensor, chance_code: Tensor) -> Dict[str, Tensor]:
        fused = self.dynamics_fusion(state, chance_code)
        latent_unnorm = self.dynamics(fused)
        return {
            "next_latent": _normalize_hidden_state(latent_unnorm),
            "unnormalized_latent": latent_unnorm,
        }


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
        is_discrete: bool = True,
        is_spatial: Optional[bool] = None,
        action_encoder: Optional[ActionEncoder] = None,
        use_bn: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.stochastic = stochastic

        # 0. Action Encoder Selection
        if action_encoder is None:
            action_encoder = get_action_encoder(
                num_actions,
                latent_dimensions,
                is_discrete=is_discrete,
                action_embedding_dim=action_embedding_dim,
                is_spatial=is_spatial,
            )

        # 1. Dynamics Strategy
        if self.stochastic:
            # Chance encoder is usually just a simple EfficientZero style embedding
            chance_encoder = get_action_encoder(
                num_chance,
                latent_dimensions,
                is_discrete=True,
                action_embedding_dim=action_embedding_dim,
                is_spatial=False,  # Chance codes are indices, not spatial
            )
            self.dynamics_pipeline = StochasticDynamics(
                latent_dimensions=latent_dimensions,
                num_chance=num_chance,
                observation_shape=observation_shape,
                dynamics_fn=dynamics_fn,
                afterstate_dynamics_fn=afterstate_dynamics_fn,
                sigma_head_fn=sigma_head_fn,
                encoder_fn=encoder_fn,
                action_encoder=action_encoder,
                chance_encoder=chance_encoder,
                use_true_chance_codes=use_true_chance_codes,
                use_bn=use_bn,
            )
        else:
            self.dynamics_pipeline = DeterministicDynamics(
                latent_dimensions=latent_dimensions,
                dynamics_fn=dynamics_fn,
                action_encoder=action_encoder,
                use_bn=use_bn,
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
            # MuZero Optimization (EfficientZero/MuZero contract):
            # The Reward head should receive the NORMALIZED state.
            # (Matches legacy MuzeroWorldModel behavior for stability)
            # TODO: maybe the head inputs should be unormalized though
            input_features = next_hidden_state

            head_out = head(
                input_features, state=head_state, is_inference=True, **kwargs
            )
            predictions[name] = head_out.training_tensor
            predictions[f"{name}_extra"] = head_out.inference_tensor

            if head_out.state:
                new_head_state.update(head_out.state)

        # The World Model packs everything it owns into one dictionary
        new_state = {"dynamics": next_hidden_state, **new_head_state}

        return WorldModelOutput(
            features=next_hidden_state,
            reward=predictions["reward_logits"],
            to_play_logits=(
                predictions["to_play_logits"]
                if "to_play_logits" in self.heads
                else None
            ),
            to_play=(
                predictions["to_play_logits_extra"]
                if "to_play_logits" in self.heads
                else None
            ),
            continuation_logits=(
                predictions["continuation_logits"]
                if "continuation_logits" in self.heads
                else None
            ),
            continuation=(
                predictions["continuation_logits_extra"]
                if "continuation_logits" in self.heads
                else None
            ),
            next_state=new_state,
            instant_reward=predictions["reward_logits_extra"],
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
            chance=res["chance_logits"],
            chance_dist=res["chance_dist"],
            next_state=new_state,
        )

    def unroll_physics(
        self,
        initial_latent_state: Tensor,
        actions: Tensor,
        initial_unnormalized_state: Optional[Tensor] = None,
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
            # Reward head should receive normalized features
            # TODO: maybe the head inputs should be unnormalized though
            input_features = current_latent

            head_out = head(
                input_features, state=current_head_state, is_inference=False, **kwargs
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
                input_features = next_latent
                # TODO: maybe the head inputs should be unnormalized though

                head_out = head(
                    input_features,
                    state=current_head_state,
                    is_inference=False,
                    **kwargs,
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
