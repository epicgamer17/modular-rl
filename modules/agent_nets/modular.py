from typing import Callable, Tuple, Dict, Any, List, Optional
import torch
from torch import nn, Tensor

from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import (
    InferenceOutput,
    LearningOutput,
    MuZeroNetworkState,
)
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.backbones.factory import BackboneFactory
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.q import QHead, DuelingQHead
from modules.heads.strategy_factory import OutputStrategyFactory
from modules.projectors.sim_siam import Projector

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
        val_strategy = OutputStrategyFactory.create(config.value_head.output_strategy)
        self.components["value_head"] = ValueHead(
            arch_config=config.arch,
            input_shape=prediction_feat_shape,
            strategy=val_strategy,
            neck_config=config.value_head.neck,
        )

        # Policy
        pol_strategy = OutputStrategyFactory.create(config.policy_head.output_strategy)
        self.components["policy_head"] = PolicyHead(
            arch_config=config.arch,
            input_shape=prediction_feat_shape,
            neck_config=config.policy_head.neck,
            strategy=pol_strategy,
        )

        # Stochastic Chance Heads (if applicable)
        if config.stochastic:
            shared_backbone_output_shape = self.components[
                "world_model"
            ].shared_backbone.output_shape
            self.components["afterstate_value_head"] = ValueHead(
                arch_config=config.arch,
                input_shape=shared_backbone_output_shape,
                strategy=val_strategy,
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
        strategy = None
        if hasattr(config.policy_head, "output_strategy"):
            strategy = OutputStrategyFactory.create(config.policy_head.output_strategy)

        self.components["policy_head"] = PolicyHead(
            arch_config=config.arch,
            input_shape=input_shape,
            neck_config=config.policy_head.neck,
            strategy=strategy,
        )

        # Value Head (Critic)
        val_strat = OutputStrategyFactory.create(config.value_head.output_strategy)
        self.components["value_head"] = ValueHead(
            arch_config=config.arch,
            input_shape=input_shape,
            strategy=val_strat,
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

        strategy = OutputStrategyFactory.create(config.head.output_strategy)

        if config.dueling:
            self.components["q_head"] = DuelingQHead(
                arch_config=config.arch,
                input_shape=current_shape,
                strategy=strategy,
                value_hidden_widths=config.head.value_hidden_widths,
                advantage_hidden_widths=config.head.advantage_hidden_widths,
                num_actions=num_actions,
                neck_config=config.head.neck,
            )
        else:
            self.components["q_head"] = QHead(
                arch_config=config.arch,
                input_shape=current_shape,
                strategy=strategy,
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
            strategy=OutputStrategyFactory.create(
                {"type": "categorical", "num_classes": num_actions}
            ),
        )

    @property
    def device(self) -> torch.device:
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        for name, module in self.components.items():
            if hasattr(module, "initialize") and callable(module.initialize):
                module.initialize(initializer)

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
            raw_value, _ = self.components["value_head"](pred_features)
            raw_policy, _, policy_dist = self.components["policy_head"](pred_features)

            expected_value = self.components["value_head"].strategy.to_expected_value(
                raw_value
            )
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
            value_logits, _ = self.components["value_head"](obs)
            expected_value = self.components["value_head"].strategy.to_expected_value(
                value_logits
            )

            return InferenceOutput(policy=policy_dist, value=expected_value)

        # ----------------------------------------
        # Rainbow DQN Logic
        # ----------------------------------------
        elif "q_head" in self.components:
            x = self.components["feature_block"](obs)
            Q = self.components["q_head"](x)

            q_vals = self.components["q_head"].strategy.to_expected_value(Q)
            state_value = q_vals.max(dim=-1)[0]

            return InferenceOutput(
                value=state_value,
                q_values=q_vals,
                policy=self.components["q_head"].strategy.get_distribution(Q),
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

    def learner_inference(self, batch: Dict[str, Any]) -> LearningOutput:
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
            raw_values, _ = self.components["value_head"](pred_features)
            raw_policies, _, _ = self.components["policy_head"](pred_features)

            raw_values = raw_values.view(B, T_plus_1, -1)
            raw_policies = raw_policies.view(B, T_plus_1, -1)

            latents_afterstates = None
            stochastic_chance_logits = None
            stochastic_chance_values = None
            chance_encoder_embeddings = None

            if (
                getattr(self.config, "stochastic", False)
                and physics_output.latents_afterstates is not None
            ):
                latents_afterstates = physics_output.latents_afterstates
                stacked_backbone_features = physics_output.afterstate_backbone_features
                B_as, T_plus_1_as = stacked_backbone_features.shape[:2]
                flat_backbone = stacked_backbone_features.view(
                    B_as * T_plus_1_as, *stacked_backbone_features.shape[2:]
                )

                raw_chance_values, _ = self.components["afterstate_value_head"](
                    flat_backbone
                )
                stochastic_chance_values = raw_chance_values.view(B_as, T_plus_1_as, -1)
                stochastic_chance_logits = physics_output.chance_logits
                chance_encoder_embeddings = physics_output.chance_encoder_embeddings

            return LearningOutput(
                values=raw_values,
                policies=raw_policies,
                rewards=physics_output.rewards,
                to_plays=physics_output.to_plays,
                latents=stacked_latents,
                latents_afterstates=latents_afterstates,
                chance_logits=stochastic_chance_logits,
                chance_values=stochastic_chance_values,
                chance_encoder_embeddings=chance_encoder_embeddings,
            )

        # ----------------------------------------
        # PPO Logic
        # ----------------------------------------
        elif "policy_head" in self.components and "value_head" in self.components:
            policy_logits, _, _ = self.components["policy_head"](initial_observation)
            value_logits, _ = self.components["value_head"](initial_observation)

            return LearningOutput(
                values=value_logits,
                policies=policy_logits,
            )

        # ----------------------------------------
        # Rainbow DQN Logic
        # ----------------------------------------
        elif "q_head" in self.components:
            # 1. Online Inference at s_t
            x = self.components["feature_block"](initial_observation)
            Q = self.components["q_head"](x)
            q_vals = self.components["q_head"].strategy.to_expected_value(Q)

            # Extract Q-values for taken actions (required for loss/priority)
            actions = batch.get("actions")
            q_taken = None
            if actions is not None:
                actions = actions.to(self.device).long()
                q_taken = q_vals[
                    torch.arange(q_vals.shape[0], device=self.device), actions
                ]

            # 2. Next State Evaluations (Decoupled Unrolling)
            next_obs = batch.get("next_observations")
            next_q_vals = None
            target_next_q_vals = None
            next_q_logits = None
            target_next_q_logits = None

            if next_obs is not None:
                next_obs = next_obs.to(self.device).float()
                with torch.no_grad():
                    # Evaluate online network on s_{t+1} (for Double DQN)
                    x_next = self.components["feature_block"](next_obs)
                    Q_next = self.components["q_head"](x_next)
                    next_q_vals = self.components["q_head"].strategy.to_expected_value(
                        Q_next
                    )
                    next_q_logits = Q_next

                    # Evaluate target network on s_{t+1} (if available)
                    if (
                        hasattr(self, "target_network")
                        and self.target_network is not None
                    ):
                        target_next_out = self.target_network.learner_inference(
                            {"observations": next_obs}
                        )
                        target_next_q_vals = target_next_out.q_values
                        target_next_q_logits = target_next_out.q_logits

            return LearningOutput(
                values=q_taken.unsqueeze(-1) if q_taken is not None else None,
                q_values=q_vals,
                q_logits=Q,
                next_q_values=next_q_vals,
                target_q_values=target_next_q_vals,
                next_q_logits=next_q_logits,
                target_q_logits=target_next_q_logits,
            )

        # ----------------------------------------
        # Supervised/Imitation Logic
        # ----------------------------------------
        elif "policy_head" in self.components and "value_head" not in self.components:
            # SL path
            x = initial_observation
            if "feature_block" in self.components:
                x = self.components["feature_block"](initial_observation)
            logits, _, _ = self.components["policy_head"](x)
            return LearningOutput(policies=logits)

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

        raw_value, _ = self.components["value_head"](pred_features)
        _, _, policy_dist = self.components["policy_head"](pred_features)

        expected_value = self.components["value_head"].strategy.to_expected_value(
            raw_value
        )
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

        raw_value, _ = self.components["afterstate_value_head"](shared_features)
        network_state_after = MuZeroNetworkState(
            dynamics=wm_output.afterstate_features,
            wm_memory=network_state.wm_memory,
        )

        chance_policy = self.components[
            "world_model"
        ].sigma_head.strategy.get_distribution(chance_logits)
        return InferenceOutput(
            network_state=network_state_after,
            value=self.components["afterstate_value_head"].strategy.to_expected_value(
                raw_value
            ),
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
