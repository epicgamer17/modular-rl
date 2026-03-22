from typing import Callable, Tuple, Dict, Any, List, Optional, Union
import torch
from torch import nn, Tensor

from modules.models.inference_output import (
    InferenceOutput,
)
from modules.models.world_model import WorldModel
from modules.backbones.factory import BackboneFactory
from modules.backbones.recurrent import RecurrentBackbone
from modules.backbones.transformer import TransformerBackbone
from modules.heads.factory import HeadFactory
from modules.projectors.sim_siam import Projector
from agents.learner.losses.shape_validator import ShapeValidator
from agents.learner.losses.representations import get_representation


class AgentNetwork(nn.Module):
    """
    The absolute center of the framework. It acts as the Switchboard between
    the RL System and the PyTorch Sub-modules.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        arch_config: Any,
        representation_config: Optional[Any] = None,
        world_model_config: Optional[Any] = None,
        prediction_backbone_config: Optional[Any] = None,
        heads_config: Dict[str, Any] = None,
        projector_config: Optional[Any] = None,
        stochastic: bool = False,
        num_players: int = 1,
        num_chance_codes: int = 0,
        validator_params: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.arch_config = arch_config
        self.stochastic = stochastic

        # --- DYNAMIC ASSEMBLY ---
        self.components = nn.ModuleDict()
        self.components["behavior_heads"] = nn.ModuleDict()

        # 1. Representation Phase (The Encoder)
        if representation_config is not None:
            self.components["representation"] = BackboneFactory.create(
                representation_config, input_shape
            )
            current_head_input_shape = self.components["representation"].output_shape
        else:
            ident = nn.Identity()
            ident.output_shape = input_shape
            self.components["representation"] = ident
            current_head_input_shape = input_shape

        self.latent_dim = current_head_input_shape

        # 2. Environment Phase (The Physics Engine)
        if world_model_config is not None:
            self.components["world_model"] = WorldModel(
                config=world_model_config,
                latent_dimensions=self.latent_dim,
                num_actions=num_actions,
                env_heads=world_model_config.env_heads,
            )

        # 3. Behavior Phase: Temporal Memory (Backbones)
        if (
            prediction_backbone_config is not None
            and "world_model" not in self.components
        ):
            backbone = BackboneFactory.create(
                prediction_backbone_config, current_head_input_shape
            )
            if isinstance(backbone, (RecurrentBackbone, TransformerBackbone)):
                self.components["memory_core"] = backbone
                current_head_input_shape = backbone.output_shape
            else:
                raise ValueError(
                    "prediction_backbone should be an RNN/Transformer. For spatial embedding, use representation_backbone."
                )

        # 3. Behavior Phase: Behavioral Heads (Policy, Value, Q, etc.)
        if heads_config:
            for head_name, head_config in heads_config.items():
                if head_config is None:
                    continue

                self.components["behavior_heads"][head_name] = HeadFactory.create(
                    head_config,
                    arch_config=arch_config,
                    input_shape=current_head_input_shape,
                    num_actions=self.num_actions,
                    num_players=num_players,
                    num_chance_codes=num_chance_codes,
                    name=head_name,
                )

        if projector_config is not None:
            hidden_state_shape = current_head_input_shape
            self.flat_hidden_dim = torch.Size(hidden_state_shape).numel()
            self.components["projector"] = Projector(
                self.flat_hidden_dim, projector_config
            )

        # Initialize Validator with passed params
        self.validator = ShapeValidator(**validator_params if validator_params else {})

    def initialize(
        self, initializer: Optional[Callable[[Tensor], None]] = None
    ) -> None:
        """Unified initialization for all components."""
        if initializer is None:
            return

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    initializer(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def reset_noise(self) -> None:
        """Resamples NoisyNet parameters across all configured components."""
        for module in self.components.values():
            if hasattr(module, "reset_noise"):
                module.reset_noise()
            if isinstance(module, nn.ModuleDict):
                for sub in module.values():
                    if hasattr(sub, "reset_noise"):
                        sub.reset_noise()

    def _apply_spatial_temporal(
        self, tensor: Tensor, B: int, T: int, state: Any = None
    ) -> Tuple[Tensor, Any]:
        flat_x = tensor.flatten(0, 1)

        next_state = None
        if "memory_core" in self.components:
            seq_x = flat_x.view(B, T, -1)
            seq_x, next_state = self.components["memory_core"](seq_x, state=state)
            return seq_x.flatten(0, 1), next_state

        return flat_x, next_state

    def obs_inference(
        self, obs: Tensor, recurrent_state: Dict[str, Tensor] = None, **kwargs
    ) -> InferenceOutput:
        """
        Root inference for initial observation (e.g. MCTS root or Actor step).
        """
        assert isinstance(obs, Tensor), "AgentNetwork strictly expects PyTorch Tensors."
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        latent = self.components["representation"](obs)

        wm_output = None

        # 2. Feature & Memory Phase
        B_val = latent.shape[0]
        features, next_h = self._apply_spatial_temporal(
            latent.unsqueeze(1), B_val, 1, state=recurrent_state
        )

        outputs = {}
        network_state = dict(recurrent_state) if recurrent_state else {}
        for name, head in self.components["behavior_heads"].items():
            head_out = head(features, state=network_state, **kwargs)
            outputs[name] = head_out.inference_tensor
            network_state.update(head_out.state)

        if wm_output:
            network_state.update(wm_output.next_state)
        elif "world_model" in self.components:
            # Fallback for root: if no wm_output (e.g. from observation),
            # we manually pack the root latent.
            network_state["dynamics"] = latent

        network_state.update(next_h)

        q_vals = outputs.get("q_logits")
        state_value = (
            q_vals.max(dim=-1)[0] if q_vals is not None else outputs.get("state_value")
        )

        return InferenceOutput(
            recurrent_state=network_state,
            value=state_value,
            policy=outputs.get("policy_logits"),
            q_values=q_vals,
            reward=None,
            to_play=wm_output.to_play if wm_output else None,
            extras={},
        )

    def learner_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified router for batch unrolls during learning."""
        obs = batch["observations"].to(self.device).float()

        # 1. Root Latent (Backbones handle sequence dimensions automatically)
        root_latent = self.components["representation"](obs)

        # 2. Environment & Temporal Phase
        if "world_model" in self.components:
            # If obs was [B, T, ...], root_latent is [B, T, *D], and we only unroll from Step 0.
            # If obs was [B, ...], root_latent is [B, *D], and it's the root.
            is_seq = root_latent.dim() > (len(self.input_shape) + 1)
            initial_latent_state = root_latent[:, 0] if is_seq else root_latent

            env_results = self.components["world_model"].unroll_physics(
                initial_latent_state=initial_latent_state,
                actions=batch["actions"],
                encoder_inputs=batch.get("chance_encoder_inputs"),
                true_chance_codes=batch.get("chance_codes"),
            )
            latents = env_results["latents"]

            # Extract afterstates immediately if they exist
            flat_as = env_results.get("latents_afterstates")
            if flat_as is not None:
                flat_as = flat_as.flatten(0, 1)
        else:
            env_results = {}
            # If not using a world model, ensure latents has Batch/Time [B, T, *D]
            latents = (
                root_latent.unsqueeze(1)
                if root_latent.dim() == (len(self.input_shape) + 1)
                else root_latent
            )
            flat_as = None

        B, T = latents.shape[:2]
        flat_memory, _ = self._apply_spatial_temporal(latents, B, T)

        # 3. The Feature Pool (Strict Contract, No Fallbacks!)
        feature_pool = {"default": flat_memory}
        if flat_as is not None:
            feature_pool["afterstate"] = flat_as

        # 4. Behavior Heads Phase
        flat_mask = batch.get("action_masks")
        if flat_mask is not None:
            flat_mask = flat_mask.flatten(0, 1)

        behavior_results = {}
        for name, head in self.components["behavior_heads"].items():
            # STRICT ROUTING: No fallback. If config is wrong, crash loudly.
            source = head.input_source
            if source not in feature_pool:
                raise ValueError(
                    f"Head '{name}' requested '{source}' features, which are not available."
                )

            head_out = head(
                feature_pool[source],
                state=batch.get("network_state", {}),
                action_mask=flat_mask,
                **batch,
            )
            behavior_results[name] = head_out.training_tensor.view(B, T, -1)

        # Note: chance_logits assignment deleted. It merges automatically below!
        final_output = {**env_results, **behavior_results, "latents": latents}
        self.validator.validate_predictions(final_output)

        return final_output

    def hidden_state_inference(
        self,
        network_state: Dict[str, Tensor],
        action: Tensor,
        **kwargs,
    ) -> InferenceOutput:
        if "world_model" not in self.components:
            raise NotImplementedError("hidden_state_inference requires a world_model.")

        dynamics_h = network_state.get("dynamics")

        wm_output = self.components["world_model"].recurrent_inference(
            hidden_state=dynamics_h,
            action=action,
            recurrent_state=network_state,
            **kwargs,
        )
        latent = wm_output.features

        # 2. Feature & Memory Phase
        B_val = latent.shape[0]
        features, next_h = self._apply_spatial_temporal(
            latent.unsqueeze(1), B_val, 1, state=network_state
        )

        outputs = {}
        next_recurrent_state = {}
        # 3. State Propagation (Blindly update from components)
        next_recurrent_state.update(wm_output.next_state)

        for name, head in self.components["behavior_heads"].items():
            head_out = head(features, state=network_state, **kwargs)
            outputs[name] = head_out.inference_tensor
            next_recurrent_state.update(head_out.state)

        next_recurrent_state.update(next_h)

        return InferenceOutput(
            recurrent_state=next_recurrent_state,
            value=outputs.get("state_value"),
            policy=outputs.get("policy_logits"),
            reward=wm_output.instant_reward,
            to_play=wm_output.to_play,
            extras={},
        )

    def afterstate_inference(
        self, recurrent_state: Dict[str, Tensor], action: Tensor
    ) -> InferenceOutput:
        if "world_model" in self.components or not self.stochastic:
            raise NotImplementedError(
                "afterstate_inference requires a stochastic world_model."
            )

        dynamics_h = recurrent_state.get("dynamics")

        wm_output = self.components["world_model"].afterstate_recurrent_inference(
            {"dynamics": dynamics_h}, action
        )

        afterstate_latent = wm_output.afterstate_features

        afterstate_head = self.components["behavior_heads"].get("afterstate_value")
        recurrent_state_after = dict(recurrent_state)
        recurrent_state_after.update(wm_output.next_state)

        head_out_as = afterstate_head(
            afterstate_latent,
            state=recurrent_state,
            afterstate_features=afterstate_latent,
        )
        expected_afterstate_value = head_out_as.inference_tensor
        recurrent_state_after.update(head_out_as.state)

        chance_policy = self.components[
            "world_model"
        ].sigma_head.representation.to_inference(wm_output.chance)

        return InferenceOutput(
            recurrent_state=recurrent_state_after,
            value=expected_afterstate_value,
            policy=chance_policy,
            chance=chance_policy,
            reward=None,
        )

    def compile(self, mode: str = "default", fullgraph: bool = False) -> None:
        if self.device.type == "mps":
            print("Skipping torch.compile on Apple Silicon (MPS).")
            return

        self.obs_inference = torch.compile(
            self.obs_inference, mode=mode, fullgraph=fullgraph
        )
        self.learner_inference = torch.compile(
            self.learner_inference, mode=mode, fullgraph=fullgraph
        )

        if "world_model" in self.components:
            self.hidden_state_inference = torch.compile(
                self.hidden_state_inference, mode=mode, fullgraph=fullgraph
            )
            if self.stochastic:
                self.afterstate_inference = torch.compile(
                    self.afterstate_inference, mode=mode, fullgraph=fullgraph
                )

    def project(self, hidden_state: Tensor, grad=True) -> Tensor:
        if "projector" not in self.components:
            raise NotImplementedError("Projector not configured for this architecture.")

        original_shape = hidden_state.shape
        flat_hidden = hidden_state.reshape(-1, self.flat_hidden_dim)
        proj = self.components["projector"].projection(flat_hidden)

        if grad:
            proj = self.components["projector"].projection_head(proj)
        else:
            proj = proj.detach()

        num_latent_dims = len(self.latent_dim)
        new_shape = list(original_shape[:-num_latent_dims]) + [proj.shape[-1]]
        return proj.reshape(new_shape)
