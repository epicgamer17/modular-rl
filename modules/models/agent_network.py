from typing import Callable, Tuple, Dict, Any, List, Optional, Union
import torch
from torch import nn, Tensor

from modules.utils import kernel_initializer_wrapper
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
        config: Any,
        input_shape: Tuple[int, ...],
        num_actions: int,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        self.num_actions = num_actions

        # --- DYNAMIC ASSEMBLY ---
        self.components = nn.ModuleDict()
        self.components["behavior_heads"] = nn.ModuleDict()

        # 1. Environment Phase (The Physics Engine)
        if config.world_model is not None:
            from modules.models.world_model import WorldModel
            world_model_cls = kwargs.get("world_model_cls", WorldModel)
            self.components["world_model"] = world_model_cls(
                config, input_shape, num_actions
            )
            current_head_input_shape = self.components[
                "world_model"
            ].representation.output_shape
        else:
            current_head_input_shape = input_shape

        # 2. Behavior Phase: Feature Extraction (Backbones)
        if config.prediction_backbone is not None:
            backbone = BackboneFactory.create(config.prediction_backbone, current_head_input_shape)
            if isinstance(backbone, (RecurrentBackbone, TransformerBackbone)):
                self.components["memory_core"] = backbone
            else:
                self.components["feature_extractor"] = backbone
            current_head_input_shape = backbone.output_shape

        # 3. Behavior Phase: Behavioral Heads (Policy, Value, Q, etc.)
        for head_name, head_config in config.heads.items():
            if head_config is None:
                continue

            # Generate representation dynamically if the head config specifies an output strategy
            # Note: Some heads (like Q) might need special handling or pre-baked representations.
            # We default to standard representation generation if 'output_strategy' is present.
            rep = None
            if hasattr(head_config, "output_strategy") and head_config.output_strategy is not None:
                from agents.learner.losses.representations import get_representation
                rep = get_representation(head_config.output_strategy)

            # Route to correct input shape (Afterstate vs Root)
            # This is currently the only notable "bleed" of logic. 
            # If a head is an 'afterstate' head, it takes shared_backbone features.
            head_input_shape = current_head_input_shape
            if head_name == "afterstate_value" and "world_model" in self.components:
                head_input_shape = self.components["world_model"].shared_backbone.output_shape

            self.components["behavior_heads"][head_name] = HeadFactory.create(
                head_config,
                arch_config=config.arch,
                input_shape=head_input_shape,
                num_actions=self.num_actions,
                num_players=getattr(config.game, "num_players", 1),
                num_chance_codes=getattr(config, "num_chance", 0),
                representation=rep,
            )

        # 4. Fallback: Identity Policy Head (Legacy support for backbone-only tests)
        if len(self.components["behavior_heads"]) == 0 and config.prediction_backbone is not None:
            from configs.modules.backbones.factory import BackboneConfigFactory
            from agents.learner.losses.representations import get_representation
            from modules.heads.policy import PolicyHead
            
            self.components["behavior_heads"]["policy_logits"] = PolicyHead(
                arch_config=config.arch,
                input_shape=current_head_input_shape,
                neck_config=BackboneConfigFactory.create({"type": "identity"}),
                representation=get_representation(
                    {"type": "classification", "num_classes": num_actions}
                ),
            )

        if config.projector is not None:
            hidden_state_shape = self.components[
                "world_model"
            ].representation.output_shape
            self.flat_hidden_dim = torch.Size(hidden_state_shape).numel()
            self.components["projector"] = Projector(self.flat_hidden_dim, config)

    def _pack_recurrent_state(
        self, 
        dynamics_latent: Optional[Tensor],
        wm_head_state: Optional[Dict[str, Any]],
        backbone_h: Optional[Union[Tensor, Tuple[Tensor, Tensor]]]
    ) -> Dict[str, Tensor]:
        """Strictly packs internal states into a flat {str: Tensor} dict."""
        state = {}
        if dynamics_latent is not None:
            state["dynamics"] = dynamics_latent
            
        if wm_head_state:
            for head_name, h in wm_head_state.items():
                if h is not None:
                    if isinstance(h, (tuple, list)):
                        for i, t in enumerate(h):
                            state[f"wm_{head_name}_{i}"] = t
                    else:
                        state[f"wm_{head_name}"] = h
                        
        if backbone_h is not None:
            if isinstance(backbone_h, (tuple, list)):
                for i, t in enumerate(backbone_h):
                    state[f"backbone_{i}"] = t
            else:
                state["backbone"] = backbone_h
        return state

    def _unpack_recurrent_state(
        self, 
        state: Dict[str, Tensor]
    ) -> Tuple[Optional[Tensor], Dict[str, Any], Any]:
        """Strictly unpacks flat {str: Tensor} dict back into internal concepts."""
        dynamics = state.get("dynamics")
        
        wm_head_state = {}
        if "world_model" in self.components:
            for head_name in self.components["world_model"].heads.keys():
                if f"wm_{head_name}" in state:
                    wm_head_state[head_name] = state[f"wm_{head_name}"]
                else:
                    i = 0
                    tuple_h = []
                    while f"wm_{head_name}_{i}" in state:
                        tuple_h.append(state[f"wm_{head_name}_{i}"])
                        i += 1
                    if tuple_h:
                        wm_head_state[head_name] = tuple(tuple_h)
        
        backbone_h = None
        if "backbone" in state:
            backbone_h = state["backbone"]
        else:
            i = 0
            tuple_bb = []
            while f"backbone_{i}" in state:
                tuple_bb.append(state[f"backbone_{i}"])
                i += 1
            if tuple_bb:
                backbone_h = tuple(tuple_bb)
                
        return dynamics, wm_head_state, backbone_h

    def initialize(
        self, initializer: Optional[Union[Callable[[Tensor], None], str]] = None
    ) -> None:
        """Unified initialization for all components."""
        init_fn = kernel_initializer_wrapper(initializer)
        if init_fn is None:
            return

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    init_fn(m.weight)
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

    def obs_inference(self, obs: Tensor) -> InferenceOutput:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        if "world_model" in self.components:
            wm_output = self.components["world_model"].initial_inference(obs)
            latent = wm_output.features
        else:
            latent = obs
            wm_output = None

        features = (
            self.components["feature_extractor"](latent)
            if "feature_extractor" in self.components
            else latent
        )

        if "memory_core" in self.components:
            seq_features = features.unsqueeze(1)
            seq_memory, next_h = self.components["memory_core"](seq_features, None)
            features = seq_memory.squeeze(1)
        else:
            next_h = None

        outputs = {}
        for name, head in self.components["behavior_heads"].items():
            out, *rest = head(features)
            outputs[name] = rest[1] if len(rest) >= 2 else out

        network_state = self._pack_recurrent_state(
            dynamics_latent=latent if "world_model" in self.components else None,
            wm_head_state=wm_output.head_state if wm_output else None,
            backbone_h=next_h,
        )

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
        
        # 1. Environment Phase
        if "world_model" in self.components:
            # Note: world_model unrolls physics across unroll_steps
            env_results = self.components["world_model"].unroll_physics(
                initial_latent_state=self.components["world_model"].initial_inference(obs).features,
                actions=batch["actions"],
                encoder_inputs=(
                    torch.cat([batch["unroll_observations"][:, :-1], batch["unroll_observations"][:, 1:]], dim=2)
                    if getattr(self.config, "stochastic", False) else None
                ),
                true_chance_codes=batch.get("chance_codes"),
                target_observations=batch.get("unroll_observations"),
            )
            latents = env_results["latents"]
        else:
            env_results = {}
            latents = obs

        B, T = latents.shape[:2]
        flat_latents = latents.flatten(0, 1)

        # 2. Spatial Phase (Feature Extraction)
        if "feature_extractor" in self.components:
            flat_features = self.components["feature_extractor"](flat_latents)
        else:
            flat_features = flat_latents

        # 3. Temporal Phase (RNN Boundary)
        if "memory_core" in self.components:
            seq_features = flat_features.view(B, T, -1)
            seq_memory, _ = self.components["memory_core"](seq_features)
            flat_memory = seq_memory.flatten(0, 1)
        else:
            flat_memory = flat_features

        # 4. Behavior Heads Phase
        behavior_results = {}
        for name, head in self.components["behavior_heads"].items():
            # Skip afterstate value head as it has its own routing path
            if name == "afterstate_value":
                continue
            flat_out, _ = head(flat_memory)
            # Use exact head names as keys. Zero string manipulation.
            behavior_results[name] = flat_out.view(B, T, -1)

        # 5. Stochastic Afterstate Phase (Exception Case)
        if "world_model" in self.components and getattr(self.config, "stochastic", False):
            as_features = env_results["afterstate_backbone_features"]
            flat_as = as_features.flatten(0, 1)
            raw_as_values, _ = self.components["behavior_heads"]["afterstate_value"](flat_as)

            B_as, K_as = as_features.shape[:2]
            as_values = raw_as_values.view(B_as, K_as, -1)
            # Prepend zero value for root
            root_as_values = torch.zeros(B_as, 1, *as_values.shape[2:], device=self.device)
            behavior_results["afterstate_values"] = torch.cat([root_as_values, as_values], dim=1)
            behavior_results["chance_logits"] = env_results.get("chance_logits")

        # Merge and Validate
        final_output = {**env_results, **behavior_results, "latents": latents}
        ShapeValidator(self.config).validate_predictions(final_output)
        return final_output

    def hidden_state_inference(self, network_state: Dict[str, Tensor], action: Tensor) -> InferenceOutput:
        if "world_model" not in self.components:
            raise NotImplementedError("hidden_state_inference requires a world_model.")

        dynamics_h, wm_head_state, backbone_h = self._unpack_recurrent_state(network_state)

        wm_output = self.components["world_model"].recurrent_inference(
            hidden_state=dynamics_h,
            action=action,
            recurrent_state=wm_head_state,
        )
        latent = wm_output.features

        features = (
            self.components["feature_extractor"](latent)
            if "feature_extractor" in self.components
            else latent
        )

        if "memory_core" in self.components:
            seq_features = features.unsqueeze(1)
            seq_memory, next_h = self.components["memory_core"](seq_features, backbone_h)
            features = seq_memory.squeeze(1)
        else:
            next_h = None

        outputs = {}
        for name, head in self.components["behavior_heads"].items():
            out, *rest = head(features)
            outputs[name] = rest[1] if len(rest) >= 2 else out

        next_recurrent_state = self._pack_recurrent_state(
            dynamics_latent=latent,
            wm_head_state=wm_output.head_state,
            backbone_h=next_h,
        )

        return InferenceOutput(
            recurrent_state=next_recurrent_state,
            value=outputs.get("state_value"),
            policy=outputs.get("policy_logits"),
            reward=wm_output.instant_reward,
            to_play=wm_output.to_play,
            extras={},
        )

    def afterstate_inference(self, recurrent_state: Dict[str, Tensor], action: Tensor) -> InferenceOutput:
        if "world_model" not in self.components or not getattr(self.config, "stochastic", False):
            raise NotImplementedError("afterstate_inference requires a stochastic world_model.")

        dynamics_h, wm_head_state, backbone_h = self._unpack_recurrent_state(recurrent_state)

        wm_output = self.components["world_model"].afterstate_recurrent_inference(
            {"dynamics": dynamics_h}, action
        )
        
        shared_backbone_features = wm_output.features
        _, *rest = self.components["behavior_heads"]["afterstate_value"](shared_backbone_features)
        expected_afterstate_value = rest[-1] if rest else None

        recurrent_state_after = self._pack_recurrent_state(
            dynamics_latent=wm_output.afterstate_features,
            wm_head_state=wm_head_state,
            backbone_h=backbone_h,
        )

        chance_policy = self.components["world_model"].sigma_head.representation.to_inference(wm_output.chance)

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

        self.obs_inference = torch.compile(self.obs_inference, mode=mode, fullgraph=fullgraph)
        try:
            self.hidden_state_inference = torch.compile(self.hidden_state_inference, mode=mode, fullgraph=fullgraph)
        except (AttributeError, NotImplementedError):
            pass
        self.learner_inference = torch.compile(self.learner_inference, mode=mode, fullgraph=fullgraph)
        try:
            self.afterstate_inference = torch.compile(self.afterstate_inference, mode=mode, fullgraph=fullgraph)
        except (AttributeError, NotImplementedError):
            pass

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

        hidden_state_shape = self.components["world_model"].representation.output_shape
        num_latent_dims = len(hidden_state_shape)
        new_shape = list(original_shape[:-num_latent_dims]) + [proj.shape[-1]]
        return proj.reshape(new_shape)
