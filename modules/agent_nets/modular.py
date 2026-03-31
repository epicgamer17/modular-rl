from typing import Callable, Tuple, Dict, Any, List, Optional
import torch
from torch import nn, Tensor

from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import (
    InferenceOutput,
    MuZeroNetworkState,
)
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.backbones.factory import BackboneFactory
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

        # Modules injected at runtime
        self.components = nn.ModuleDict()

        config_type = getattr(config, "agent_type", None)

        if isinstance(config, MuZeroConfig) or config_type == "muzero":
            self._init_muzero(config, input_shape, num_actions, **kwargs)
        elif isinstance(config, PPOConfig) or config_type == "ppo":
            self._init_ppo(config, input_shape, num_actions, **kwargs)
        elif isinstance(config, RainbowConfig) or config_type == "rainbow":
            self._init_rainbow(config, input_shape, num_actions, **kwargs)
        elif isinstance(config, SupervisedConfig) or config_type == "supervised":
            self._init_sl(config, input_shape, num_actions, **kwargs)
        else:
            raise ValueError(
                f"Unsupported config type for ModularAgentNetwork: {type(config)}"
            )

    def _init_muzero(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int, ...],
        num_actions: int,
        **kwargs,
    ):
        # 1. The Physics Engine
        world_model_cls = kwargs.get("world_model_cls", MuzeroWorldModel)
        self.components["world_model"] = world_model_cls(
            config, input_shape, num_actions
        )

        hidden_state_shape = self.components["world_model"].representation.output_shape

        # 2. The Task-Specific Heads
        self.components["prediction_backbone"] = BackboneFactory.create(
            config.prediction_backbone, hidden_state_shape
        )
        prediction_feat_shape = self.components["prediction_backbone"].output_shape

        # Value
        val_rep = get_representation(config.value_head.output_strategy)
        self.components["value_head"] = ValueHead(
            arch_config=config.arch,
            input_shape=prediction_feat_shape,
            representation=val_rep,
            neck_config=config.value_head.neck,
        )

        # Policy
        pol_rep = get_representation(config.policy_head.output_strategy)
        self.components["policy_head"] = PolicyHead(
            arch_config=config.arch,
            input_shape=prediction_feat_shape,
            neck_config=config.policy_head.neck,
            representation=pol_rep,
        )

        # Stochastic Chance Heads (if applicable)
        if config.stochastic:
            shared_backbone_output_shape = self.components[
                "world_model"
            ].shared_backbone.output_shape
            self.components["afterstate_value_head"] = ValueHead(
                arch_config=config.arch,
                input_shape=shared_backbone_output_shape,
                representation=val_rep,
                neck_config=config.value_head.neck,
            )

        # 3. Efficient Zero Projector
        self.flat_hidden_dim = torch.Size(hidden_state_shape).numel()
        self.components["projector"] = Projector(self.flat_hidden_dim, config)

    def _init_ppo(
        self,
        config: PPOConfig,
        input_shape: Tuple[int, ...],
        num_actions: int,
        **kwargs,
    ):
        # Policy Head (Actor)
        pol_rep = None
        if hasattr(config.policy_head, "output_strategy"):
            pol_rep = get_representation(config.policy_head.output_strategy)

        self.components["policy_head"] = PolicyHead(
            arch_config=config.arch,
            input_shape=input_shape,
            neck_config=config.policy_head.neck,
            representation=pol_rep,
        )

        # Value Head (Critic)
        val_rep = get_representation(config.value_head.output_strategy)
        self.components["value_head"] = ValueHead(
            arch_config=config.arch,
            input_shape=input_shape,
            representation=val_rep,
            neck_config=config.value_head.neck,
        )

    def _init_rainbow(
        self,
        config: RainbowConfig,
        input_shape: Tuple[int, ...],
        num_actions: int,
        **kwargs,
    ):
        # Feature Extraction
        self.components["feature_block"] = BackboneFactory.create(
            config.backbone, input_shape
        )
        current_shape = self.components["feature_block"].output_shape

        representation = get_representation(config.head.output_strategy)

        if config.dueling:
            self.components["q_head"] = DuelingQHead(
                arch_config=config.arch,
                input_shape=current_shape,
                representation=representation,
                value_hidden_widths=config.head.value_hidden_widths,
                advantage_hidden_widths=config.head.advantage_hidden_widths,
                num_actions=num_actions,
                neck_config=config.head.neck,
            )
        else:
            self.components["q_head"] = QHead(
                arch_config=config.arch,
                input_shape=current_shape,
                representation=representation,
                hidden_widths=config.head.hidden_widths,
                num_actions=num_actions,
                neck_config=config.head.neck,
            )

    def _init_sl(
        self,
        config: SupervisedConfig,
        input_shape: Tuple[int, ...],
        num_actions: int,
        **kwargs,
    ):
        """Initializes components for supervised learning/imitation."""
        # Simple policy head based on backbone
        self.components["feature_block"] = BackboneFactory.create(
            config.backbone, input_shape
        )
        current_shape = self.components["feature_block"].output_shape

        # We use a basic policy head. SupervisedConfig might not have a full PolicyHeadConfig,
        # but we can construct one or just use a simple linear layer.
        # For consistency with other modular nets, we'll try to use a PolicyHead if possible.
        from configs.modules.heads.policy import PolicyHeadConfig
        from configs.modules.backbones.factory import BackboneConfigFactory

        # Create a default PolicyHeadConfig if not in SupervisedConfig
        neck_config = BackboneConfigFactory.create({"type": "identity"})
        pol_head_config = PolicyHeadConfig(
            {"neck": {"type": "identity"}, "output_strategy": {"type": "categorical"}}
        )

        self.components["policy_head"] = PolicyHead(
            arch_config=config.arch,
            input_shape=current_shape,
            neck_config=neck_config,
            representation=get_representation(
                {"type": "classification", "num_classes": num_actions}
            ),
        )

    @property
    def device(self) -> torch.device:
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

    def reset_noise(self) -> None:
        """Resamples NoisyNet parameters across all configured components."""
        for name, module in self.components.items():
            if hasattr(module, "reset_noise"):
                module.reset_noise()

    def obs_inference(self, obs: Tensor) -> InferenceOutput:
        """
        Universal Actor API: Translates raw observations based
        on the exact flow implied by the instantiated components.
        """
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if obs.dim() == len(self.input_shape):
            obs = obs.unsqueeze(0)

        # ----------------------------------------
        # MuZero Logic
        # ----------------------------------------
        if "world_model" in self.components:
            wm_output = self.components["world_model"].initial_inference(obs)
            hidden_state = wm_output.features

            pred_features = self.components["prediction_backbone"](hidden_state)
            raw_value, _, expected_value = self.components["value_head"](pred_features)
            raw_policy, _, policy_dist = self.components["policy_head"](pred_features)
            network_state = MuZeroNetworkState(
                dynamics=hidden_state,
                wm_memory=wm_output.head_state,
            )

            return InferenceOutput(
                network_state=network_state,
                value=expected_value,
                policy=policy_dist,
                reward=None,
                to_play=wm_output.to_play,
                extras={},
            )

        # ----------------------------------------
        # PPO Logic
        # ----------------------------------------
        elif "policy_head" in self.components and "value_head" in self.components:
            policy_logits, _, policy_dist = self.components["policy_head"](obs)
            value_logits, _, expected_value = self.components["value_head"](obs)

            return InferenceOutput(policy=policy_dist, value=expected_value)

        # ----------------------------------------
        # Rainbow DQN Logic
        # ----------------------------------------
        elif "q_head" in self.components:
            x = self.components["feature_block"](obs)
            Q_logits, _, policy_dist = self.components["q_head"](x)

            q_vals = self.components["q_head"].representation.to_expected_value(
                Q_logits
            )
            state_value = q_vals.max(dim=-1)[0]

            return InferenceOutput(
                value=state_value, q_values=q_vals, policy=policy_dist
            )

        # ----------------------------------------
        # Supervised/Imitation Logic
        # ----------------------------------------
        elif "policy_head" in self.components:
            # If we only have a policy head (and maybe a feature block)
            x = obs
            if "feature_block" in self.components:
                x = self.components["feature_block"](obs)

            logits, _, dist = self.components["policy_head"](x)
            return InferenceOutput(policy=dist)

        else:
            raise NotImplementedError(
                "Network components don't match any known inference pipeline."
            )

    def learner_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal Learner API: routes batch data appropriately based on components.
        """
        initial_observation = batch.get("observations")
        assert initial_observation is not None, "Batch must contain 'observations'"
        initial_observation = initial_observation.to(self.device).float()

        # ----------------------------------------
        # MuZero Logic
        # ----------------------------------------
        if "world_model" in self.components:
            actions = batch.get("actions")
            assert actions is not None, "Batch must contain 'actions' for MuZero"
            target_observations = batch.get("unroll_observations")
            target_chance_codes = batch.get("chance_codes")

            wm_output = self.components["world_model"].initial_inference(
                initial_observation
            )
            latent = wm_output.features
            head_state = wm_output.head_state

            encoder_inputs = None
            if (
                getattr(self.config, "stochastic", False)
                and target_observations is not None
            ):
                encoder_inputs = torch.cat(
                    [target_observations[:, :-1], target_observations[:, 1:]], dim=2
                )

            physics_output = self.components["world_model"].unroll_physics(
                initial_latent_state=latent,
                actions=actions,
                encoder_inputs=encoder_inputs,
                true_chance_codes=target_chance_codes,
                head_state=head_state,
                target_observations=target_observations,
            )

            stacked_latents = physics_output.latents
            B, T_plus_1 = stacked_latents.shape[:2]
            flat_latents = stacked_latents.reshape(
                B * T_plus_1, *stacked_latents.shape[2:]
            )

            pred_features = self.components["prediction_backbone"](flat_latents)
            raw_values, _, _ = self.components["value_head"](pred_features)
            raw_policies, _, _ = self.components["policy_head"](pred_features)

            raw_values = raw_values.view(B, T_plus_1, -1)
            raw_policies = raw_policies.view(B, T_plus_1, -1)

            latents_afterstates = None
            stochastic_chance_logits = None
            stochastic_chance_values = None
            chance_encoder_embeddings = None

            # --- PAD TRANSITION OUTPUTS (Rewards/Chance) ---
            # MuZero predicts rewards for transitions (k=1..K).
            # We pad index 0 with zeros to satisfy the Universal [B, K+1, ...] contract.
            dummy_reward = torch.zeros(
                B,
                1,
                *physics_output.rewards.shape[2:],
                device=self.device,
                dtype=physics_output.rewards.dtype,
            )
            padded_rewards = torch.cat([dummy_reward, physics_output.rewards], dim=1)

            if (
                getattr(self.config, "stochastic", False)
                and physics_output.latents_afterstates is not None
            ):
                latents_afterstates = physics_output.latents_afterstates
                stacked_backbone_features = physics_output.afterstate_backbone_features
                B_as, T_as = stacked_backbone_features.shape[:2]  # T_as is K
                flat_backbone = stacked_backbone_features.reshape(
                    B_as * T_as, *stacked_backbone_features.shape[2:]
                )

                raw_chance_values, _, _ = self.components["afterstate_value_head"](
                    flat_backbone
                )
                # Stochastic Values: [B, K, atoms] -> [B, K+1, atoms]
                stochastic_chance_values = raw_chance_values.view(B_as, T_as, -1)
                dummy_chance_values = torch.zeros(
                    B_as,
                    1,
                    *stochastic_chance_values.shape[2:],
                    device=self.device,
                    dtype=stochastic_chance_values.dtype,
                )
                stochastic_chance_values = torch.cat(
                    [dummy_chance_values, stochastic_chance_values], dim=1
                )

                # Stochastic Chance Logits: [B, K, num_chance] -> [B, K+1, num_chance]
                stochastic_chance_logits = physics_output.chance_logits
                dummy_chance_logits = torch.zeros(
                    B_as,
                    1,
                    *stochastic_chance_logits.shape[2:],
                    device=self.device,
                    dtype=stochastic_chance_logits.dtype,
                )
                stochastic_chance_logits = torch.cat(
                    [dummy_chance_logits, stochastic_chance_logits], dim=1
                )

                # Stochastic Chance Encoder Embeddings: [B, K, dim] -> [B, K+1, dim]
                chance_encoder_embeddings = physics_output.chance_encoder_embeddings
                dummy_chance_embeddings = torch.zeros(
                    B_as,
                    1,
                    *chance_encoder_embeddings.shape[2:],
                    device=self.device,
                    dtype=chance_encoder_embeddings.dtype,
                )
                chance_encoder_embeddings = torch.cat(
                    [dummy_chance_embeddings, chance_encoder_embeddings], dim=1
                )

            output = {
                "values": raw_values,
                "policies": raw_policies,
                "rewards": padded_rewards,
                "to_plays": physics_output.to_plays,
                "latents": stacked_latents,
            }

            if (
                getattr(self.config, "stochastic", False)
                and latents_afterstates is not None
            ):
                output["latents_afterstates"] = latents_afterstates
                output["chance_logits"] = stochastic_chance_logits
                output["chance_values"] = stochastic_chance_values
                output["chance_encoder_embeddings"] = chance_encoder_embeddings

            # --- SHAPE VALIDATION ---
            # MuZero unrolls k steps, so T = k + 1
            expected_T = getattr(self.config, "unroll_steps", 0) + 1
            assert (
                output["values"].ndim >= 2 and output["values"].shape[1] == expected_T
            ), f"MuZero values shape mismatch: {output['values'].shape}"
            assert (
                output["policies"].ndim >= 3
                and output["policies"].shape[1] == expected_T
            ), f"MuZero policies shape mismatch: {output['policies'].shape}"

            # --- STRICT SYSTEM-WIDE VALIDATION ---
            ShapeValidator(
                minibatch_size=self.config.minibatch_size,
                unroll_steps=getattr(self.config, "unroll_steps", 0),
                num_actions=self.num_actions,
                atom_size=getattr(self.config, "atom_size", 1),
                support_range=getattr(self.config, "support_range", None),
            ).validate_predictions(output)

            return output

        # ----------------------------------------
        # PPO Logic
        # ----------------------------------------
        elif "policy_head" in self.components and "value_head" in self.components:
            policy_logits, _, _ = self.components["policy_head"](initial_observation)
            value_logits, _, _ = self.components["value_head"](initial_observation)

            output = {
                "values": value_logits.unsqueeze(1),
                "policies": policy_logits.unsqueeze(1),
            }

            # --- SHAPE VALIDATION ---
            # PPO is single-step by default in learner_inference
            assert (
                output["values"].ndim == 3
                and output["values"].shape[1] == 1
                and output["values"].shape[2] == 1
            )
            assert (
                output["policies"].ndim == 3 and output["policies"].shape[1] == 1
            ), f"PPO policies shape mismatch: {output['policies'].shape}"

            # --- STRICT SYSTEM-WIDE VALIDATION ---
            ShapeValidator(
                minibatch_size=self.config.minibatch_size,
                unroll_steps=getattr(self.config, "unroll_steps", 0),
                num_actions=self.num_actions,
                atom_size=getattr(self.config, "atom_size", 1),
                support_range=getattr(self.config, "support_range", None),
            ).validate_predictions(output)

            return output

        # ----------------------------------------
        # Rainbow DQN Logic
        # ----------------------------------------
        elif "q_head" in self.components:
            # 1. Online Inference at s_t
            x = self.components["feature_block"](initial_observation)
            Q_logits, _, _ = self.components["q_head"](x)
            q_vals = self.components["q_head"].representation.to_expected_value(
                Q_logits
            )

            output = {
                "q_values": q_vals.unsqueeze(1),
                "q_logits": Q_logits.unsqueeze(1),
            }

            # --- SHAPE VALIDATION ---
            assert (
                output["q_values"].ndim == 3 and output["q_values"].shape[1] == 1
            ), f"Rainbow q_values shape mismatch: {output['q_values'].shape}"
            assert (
                output["q_logits"].ndim >= 3 and output["q_logits"].shape[1] == 1
            ), f"Rainbow q_logits shape mismatch: {output['q_logits'].shape}"

            # --- STRICT SYSTEM-WIDE VALIDATION ---
            ShapeValidator(
                minibatch_size=self.config.minibatch_size,
                unroll_steps=getattr(self.config, "unroll_steps", 0),
                num_actions=self.num_actions,
                atom_size=getattr(self.config, "atom_size", 1),
                support_range=getattr(self.config, "support_range", None),
            ).validate_predictions(output)

            return output

        # ----------------------------------------
        # Supervised/Imitation Logic
        # ----------------------------------------
        elif "policy_head" in self.components and "value_head" not in self.components:
            # SL path
            x = initial_observation
            if "feature_block" in self.components:
                x = self.components["feature_block"](initial_observation)
            logits, _, _ = self.components["policy_head"](x)
            output = {"policies": logits.unsqueeze(1)}

            # --- SHAPE VALIDATION ---
            assert (
                output["policies"].ndim == 3 and output["policies"].shape[1] == 1
            ), f"Imitation policies shape mismatch: {output['policies'].shape}"

            # --- STRICT SYSTEM-WIDE VALIDATION ---
            ShapeValidator(
                minibatch_size=self.config.minibatch_size,
                unroll_steps=getattr(self.config, "unroll_steps", 0),
                num_actions=self.num_actions,
                atom_size=getattr(self.config, "atom_size", 1),
                support_range=getattr(self.config, "support_range", None),
            ).validate_predictions(output)

            return output

        else:
            raise NotImplementedError(
                "Network components don't match any known learner inference pipeline."
            )

    # ==========================================
    # SEARCH API (Only relevant for MuZero routing)
    # ==========================================
    def hidden_state_inference(
        self, network_state: MuZeroNetworkState, action: Tensor
    ) -> InferenceOutput:
        if "world_model" not in self.components:
            return super().hidden_state_inference(network_state, action)

        dynamics_state = network_state.dynamics
        wm_memory = network_state.wm_memory

        wm_output = self.components["world_model"].recurrent_inference(
            hidden_state=dynamics_state,
            action=action,
            recurrent_state=wm_memory,
        )

        next_hidden = wm_output.features
        pred_features = self.components["prediction_backbone"](next_hidden)

        _, _, expected_value = self.components["value_head"](pred_features)
        _, _, policy_dist = self.components["policy_head"](pred_features)
        next_network_state = MuZeroNetworkState(
            dynamics=next_hidden,
            wm_memory=wm_output.head_state,
        )

        return InferenceOutput(
            network_state=next_network_state,
            value=expected_value,
            policy=policy_dist,
            reward=wm_output.instant_reward,
            to_play=wm_output.to_play,
            extras={},
        )

    def afterstate_inference(
        self, network_state: Any, action: Tensor
    ) -> InferenceOutput:
        if "world_model" not in self.components or not getattr(
            self.config, "stochastic", False
        ):
            return super().afterstate_inference(network_state, action)

        wm_output = self.components["world_model"].afterstate_recurrent_inference(
            network_state, action
        )
        shared_features = wm_output.features
        chance_logits = wm_output.chance

        _, _, expected_afterstate_value = self.components["afterstate_value_head"](
            shared_features
        )
        network_state_after = MuZeroNetworkState(
            dynamics=wm_output.afterstate_features,
            wm_memory=network_state.wm_memory,
        )

        chance_policy = self.components[
            "world_model"
        ].sigma_head.representation.to_inference(chance_logits)
        return InferenceOutput(
            network_state=network_state_after,
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
