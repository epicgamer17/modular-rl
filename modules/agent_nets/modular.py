from typing import Callable, Tuple, Dict, Any, List, Optional
import torch
from torch import nn, Tensor

from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import (
    InferenceOutput,
)
from modules.world_models.modular_world_model import ModularWorldModel
from modules.backbones.factory import BackboneFactory
from modules.backbones.recurrent import RecurrentBackbone
from modules.backbones.transformer import TransformerBackbone
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.q import QHead, DuelingQHead
from agents.learner.losses.representations import get_representation
from modules.projectors.sim_siam import Projector
from agents.learner.losses.shape_validator import ShapeValidator

from configs.agents.muzero import MuZeroConfig
from configs.agents.ppo import PPOConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.supervised import SupervisedConfig


class ModularAgentNetwork(BaseAgentNetwork):
    """
    A universal agent network that dynamically instantiates components
    such as `world_model`, `prediction_backbone`, `policy_head`, `value_head`,
    and `q_head` based on the provided configuration.

    Data routing during inference mathematically aligns with the old separate
    architectures (MuZero, PPO, Rainbow DQN) using conditional pipeline logic
    keyed on the components actually built into `self.components`.
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
        # MuZero-style dynamics or World Model
        if hasattr(config, "representation_backbone") or hasattr(config, "world_model"):
            world_model_cls = kwargs.get("world_model_cls", ModularWorldModel)
            self.components["world_model"] = world_model_cls(
                config, input_shape, num_actions
            )
            current_head_input_shape = self.components[
                "world_model"
            ].representation.output_shape
        else:
            current_head_input_shape = input_shape

        # 2. Behavior Phase: Feature Extraction (Backbones)
        backbone_config = getattr(
            config, "prediction_backbone", getattr(config, "backbone", None)
        )

        # If backbone_config is specified, it might be a memory_core or a feature_extractor
        if backbone_config:
            backbone = BackboneFactory.create(backbone_config, current_head_input_shape)
            if isinstance(backbone, (RecurrentBackbone, TransformerBackbone)):
                self.components["memory_core"] = backbone
            else:
                self.components["feature_extractor"] = backbone
            current_head_input_shape = backbone.output_shape

        # 3. Behavior Phase: Behavioral Heads (Policy, Value, Q)
        if hasattr(config, "policy_head"):
            pol_rep = (
                get_representation(config.policy_head.output_strategy)
                if hasattr(config.policy_head, "output_strategy")
                else None
            )
            self.components["behavior_heads"]["policy_logits"] = PolicyHead(
                arch_config=config.arch,
                input_shape=current_head_input_shape,
                neck_config=config.policy_head.neck,
                representation=pol_rep,
            )

        if hasattr(config, "value_head"):
            val_rep = get_representation(config.value_head.output_strategy)
            self.components["behavior_heads"]["state_value"] = ValueHead(
                arch_config=config.arch,
                input_shape=current_head_input_shape,
                representation=val_rep,
                neck_config=config.value_head.neck,
            )

        # Handle stochastic extra heads
        if getattr(config, "stochastic", False):
            val_rep = get_representation(config.value_head.output_strategy)
            shared_backbone_output_shape = self.components[
                "world_model"
            ].shared_backbone.output_shape
            self.components["behavior_heads"]["afterstate_value"] = ValueHead(
                arch_config=config.arch,
                input_shape=shared_backbone_output_shape,
                representation=val_rep,
                neck_config=config.value_head.neck,
            )

        # Handle Rainbow-style Q Heads
        if hasattr(config, "head") and not hasattr(config, "policy_head"):
            representation = get_representation(config.head.output_strategy)
            if getattr(config, "dueling", False):
                self.components["behavior_heads"]["q_logits"] = DuelingQHead(
                    arch_config=config.arch,
                    input_shape=current_head_input_shape,
                    representation=representation,
                    value_hidden_widths=config.head.value_hidden_widths,
                    advantage_hidden_widths=config.head.advantage_hidden_widths,
                    num_actions=num_actions,
                    neck_config=config.head.neck,
                )
            else:
                self.components["behavior_heads"]["q_logits"] = QHead(
                    arch_config=config.arch,
                    input_shape=current_head_input_shape,
                    representation=representation,
                    hidden_widths=config.head.hidden_widths,
                    num_actions=num_actions,
                    neck_config=config.head.neck,
                )

        # Fallback for Supervised/Imitation (No explicit head config, just a backbone)
        if len(self.components["behavior_heads"]) == 0 and backbone_config:
            # Create a default categorical policy head for imitation
            from configs.modules.backbones.factory import BackboneConfigFactory

            self.components["behavior_heads"]["policy_logits"] = PolicyHead(
                arch_config=config.arch,
                input_shape=current_head_input_shape,
                neck_config=BackboneConfigFactory.create({"type": "identity"}),
                representation=get_representation(
                    {"type": "classification", "num_classes": num_actions}
                ),
            )

        # 4. Projector (For EfficientZero)
        if hasattr(config, "projector"):
            hidden_state_shape = self.components[
                "world_model"
            ].representation.output_shape
            self.flat_hidden_dim = torch.Size(hidden_state_shape).numel()
            self.components["projector"] = Projector(self.flat_hidden_dim, config)

    @property
    def device(self) -> torch.device:
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

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
        """
        Actor Inference (Unified Routing): latent -> feature_extractor -> memory_core -> behavior_heads
        """
        # 1. Input Processing
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        # 2. LATENT PHASE (World Model)
        if "world_model" in self.components:
            wm_output = self.components["world_model"].initial_inference(obs)
            latent = wm_output.features
        else:
            latent = obs
            wm_output = None

        # 3. BEHAVIOR PHASE: Feature Extraction
        features = (
            self.components["feature_extractor"](latent)
            if "feature_extractor" in self.components
            else latent
        )

        # 4. BEHAVIOR PHASE: Memory Core
        # During acting (T=1), memory_core gets a dummy time dimension.
        if "memory_core" in self.components:
            seq_features = features.unsqueeze(1)  # (B, 1, D)
            # Fetch backbone memory from network_state to maintain continuity
            # (Note: Actor state management happens outside this core inference)
            seq_memory, next_h = self.components["memory_core"](seq_features, None)
            features = seq_memory.squeeze(1)
        else:
            next_h = None

        # 5. BEHAVIOR PHASE: Route to Terminal Heads
        outputs = {}
        head_dicts = self.components["behavior_heads"]
        for name, head in head_dicts.items():
            out, *rest = head(features)
            # Index 1 of rest is the 3rd return value (inference/expected_value)
            outputs[name] = rest[1] if len(rest) >= 2 else out

        # 6. Assemble InferenceOutput
        # network_state persists the opaque tokens for the world model and memory core
        network_state = {
            "dynamics": latent if "world_model" in self.components else None,
            "wm_memory": wm_output.head_state if wm_output else None,
            "backbone_memory": next_h,
        }

        # Handling Q-Values (Rainbow)
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
        """
        Learner Inference (The Explicit Flat-Batch Pipeline)
        """
        obs = batch.get("observations")
        assert obs is not None, "Batch must contain 'observations'"
        obs = obs.to(self.device).float()

        # 1. ENVIRONMENT PHASE (World Model / Physics)
        if "world_model" in self.components:
            env_results = self.components["world_model"].unroll_physics(
                initial_latent_state=self.components["world_model"]
                .initial_inference(obs)
                .features,
                actions=batch["actions"],
                encoder_inputs=(
                    torch.cat(
                        [
                            batch["unroll_observations"][:, :-1],
                            batch["unroll_observations"][:, 1:],
                        ],
                        dim=2,
                    )
                    if getattr(self.config, "stochastic", False)
                    else None
                ),
                true_chance_codes=batch.get("chance_codes"),
                target_observations=batch.get("unroll_observations"),
            )
            latents = env_results["latents"]
        else:
            # For PPO/Rainbow, the environment phase is just the raw sequence
            latents = obs
            env_results = {"observations": obs}

        B, T = latents.shape[:2]

        # 2. FLATTEN: Spatial/Dense Feature Extraction
        flat_latents = latents.flatten(0, 1)
        if "feature_extractor" in self.components:
            flat_features = self.components["feature_extractor"](flat_latents)
        else:
            flat_features = flat_latents

        # 3. BEHAVIOR PHASE: Temporal Memory (Sequence Boundary)
        if "memory_core" in self.components:
            seq_features = flat_features.view(B, T, -1)
            seq_memory, _ = self.components["memory_core"](seq_features)
            flat_memory = seq_memory.flatten(0, 1)
        else:
            flat_memory = flat_features

        # 4. BEHAVIOR PHASE: Route to Terminal Heads
        behavior_results = {}
        for name, head in self.components["behavior_heads"].items():
            if name == "afterstate_value":
                continue # Handled in section 5
            flat_out, *rest = head(flat_memory)
            behavior_results[name] = flat_out.view(B, T, -1)

        # 5. STOCHASTIC ADDITIONAL HEADS (Afterstate Values)
        if "world_model" in self.components and getattr(
            self.config, "stochastic", False
        ):
            # Special handling for stochastic afterstate values
            # Physics output has 'afterstate_backbone_features' [B, K, D]
            as_features = env_results["afterstate_backbone_features"]
            flat_as = as_features.flatten(0, 1)
            raw_as_values, *rest = self.components["behavior_heads"]["afterstate_value"](
                flat_as
            )

            # Unflatten and Pad (Root step has no afterstate value)
            B_as, K_as = as_features.shape[:2]
            as_values = raw_as_values.view(B_as, K_as, -1)
            root_as_values = torch.zeros(
                B_as, 1, *as_values.shape[2:], device=self.device
            )
            behavior_results["afterstate_value"] = torch.cat(
                [root_as_values, as_values], dim=1
            )
            behavior_results["chance_logits"] = env_results.get("chance_logits")

        # 6. MERGE: One massive dictionary for the loss pipeline
        final_output = {**env_results, **behavior_results}

        final_output["latents"] = latents
        ShapeValidator(self.config).validate_predictions(final_output)
        return final_output

    # ==========================================
    # SEARCH API (Only relevant for MuZero routing)
    # ==========================================
    def hidden_state_inference(
        self, network_state: Dict[str, Any], action: Tensor
    ) -> InferenceOutput:
        """
        MCTS Routing: Step the world model and immediately pass through behavior heads.
        """
        if "world_model" not in self.components:
            return super().hidden_state_inference(network_state, action)

        wm_output = self.components["world_model"].recurrent_inference(
            hidden_state=network_state["dynamics"],
            action=action,
            recurrent_state=network_state.get("wm_memory"),
        )
        latent = wm_output.features

        # Feature Extraction
        features = (
            self.components["feature_extractor"](latent)
            if "feature_extractor" in self.components
            else latent
        )

        # Memory Core
        if "memory_core" in self.components:
            seq_features = features.unsqueeze(1)
            seq_memory, next_h = self.components["memory_core"](
                seq_features, network_state.get("backbone_memory")
            )
            features = seq_memory.squeeze(1)
        else:
            next_h = None

        # Behavior Heads
        outputs = {}
        head_dicts = self.components["behavior_heads"]
        for name, head in head_dicts.items():
            out, *rest = head(features)
            # Index 1 of rest is the 3rd return value (inference/expected_value)
            outputs[name] = rest[1] if len(rest) >= 2 else out

        next_recurrent_state = {
            "dynamics": latent,
            "wm_memory": wm_output.head_state,
            "backbone_memory": next_h,
        }

        return InferenceOutput(
            recurrent_state=next_recurrent_state,
            value=outputs.get("state_value"),
            policy=outputs.get("policy_logits"),
            reward=wm_output.instant_reward,
            to_play=wm_output.to_play,
            extras={},
        )

    def afterstate_inference(
        self, recurrent_state: Any, action: Tensor
    ) -> InferenceOutput:
        """
        MCTS Routing: Step the world model's afterstate and return afterstate value.
        """
        if "world_model" not in self.components or not getattr(
            self.config, "stochastic", False
        ):
            return super().afterstate_inference(recurrent_state, action)

        wm_output = self.components["world_model"].afterstate_recurrent_inference(
            recurrent_state, action
        )
        as_latent = wm_output.features

        # Afterstate Value prediction uses the shared backbone from the world model
        shared_backbone_features = (
            wm_output.features
        )  # Correctly returning features from shared backbone
        _, *rest = self.components["behavior_heads"]["afterstate_value"](
            shared_backbone_features
        )
        expected_afterstate_value = rest[-1] if rest else None

        recurrent_state_after = {
            "dynamics": wm_output.afterstate_features,
            "wm_memory": recurrent_state.get("wm_memory"),
        }

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

    def project(self, hidden_state: Tensor, grad=True) -> Tensor:
        """
        Projects hidden state to embedding space (For EfficientZero consistency).
        """
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
