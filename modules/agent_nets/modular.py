from typing import Callable, Tuple, Dict, Any, List, Optional
import torch
from torch import nn, Tensor

from modules.agent_nets.base import BaseAgentNetwork
from modules.world_models.inference_output import (
    InferenceOutput,
    MuZeroNetworkState,
)
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from modules.heads.q import QHead, DuelingQHead
from modules.representations import get_representation
from modules.projectors.sim_siam import Projector
from components.losses import ShapeValidator


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
        components: Dict[str, nn.Module],
        **kwargs: Any,
    ):
        """
        Initializes the ModularAgentNetwork.

        Args:
            components: Dictionary of sub-modules (e.g., 'backbone', 'policy_head').
            **kwargs: Additional metadata (ignored if no longer needed).
        """
        super().__init__()
        self.components = nn.ModuleDict(components)

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
        assert torch.is_tensor(obs), f"obs_inference expects a tensor, got {type(obs)}. Stage 2 (Tensor Routing) must happen outside the network."
        # obs typically arrives as uint8 from Stage 2. Cast to float here (Neural Preprocessing).
        obs = obs.float()

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
            x = obs
            if "backbone" in self.components:
                x = self.components["backbone"](obs)
            elif "feature_block" in self.components:
                x = self.components["feature_block"](obs)

            policy_logits, _, policy_dist = self.components["policy_head"](x)
            value_logits, _, expected_value = self.components["value_head"](x)

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

    def learner_inference(
        self, batch: Dict[str, Any], shape_validator: Optional[ShapeValidator] = None
    ) -> Dict[str, Any]:
        """
        Universal Learner API: routes batch data appropriately based on components.
        """
        initial_observation = batch.get("observations")
        assert initial_observation is not None, "Batch must contain 'observations'"
        # Stage 2 (Tensor Routing) must have moved the batch to the correct device.
        # We perform type casting here (Neural Preprocessing).
        initial_observation = initial_observation.float()

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
            stochastic = getattr(self.components["world_model"], "stochastic", False)
            if stochastic and target_observations is not None:
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
            raw_policies, _, _ = self.components["policy_head"](
                raw_values
                if "prediction_backbone" not in self.components
                else pred_features
            )
            # Note: The above logic for policy_head input depends on the architecture.
            # In MuZero, policy and value heads often share the same backbone features.
            raw_policies, _, _ = self.components["policy_head"](pred_features)

            raw_values = raw_values.view(B, T_plus_1, -1)
            raw_policies = raw_policies.view(B, T_plus_1, -1)

            latents_afterstates = None
            stochastic_chance_logits = None
            stochastic_chance_values = None
            chance_encoder_embeddings = None

            # --- PAD TRANSITION OUTPUTS (Rewards/Chance) ---
            dummy_reward = torch.zeros(
                B,
                1,
                *physics_output.rewards.shape[2:],
                device=self.device,
                dtype=physics_output.rewards.dtype,
            )
            padded_rewards = torch.cat([dummy_reward, physics_output.rewards], dim=1)

            if stochastic and physics_output.latents_afterstates is not None:
                latents_afterstates = physics_output.latents_afterstates
                stacked_backbone_features = physics_output.afterstate_backbone_features
                B_as, T_as = stacked_backbone_features.shape[:2]  # T_as is K
                flat_backbone = stacked_backbone_features.reshape(
                    B_as * T_as, *stacked_backbone_features.shape[2:]
                )

                raw_chance_values, _, _ = self.components["afterstate_value_head"](
                    flat_backbone
                )
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

            if stochastic and latents_afterstates is not None:
                output["latents_afterstates"] = latents_afterstates
                output["chance_logits"] = stochastic_chance_logits
                output["chance_values"] = stochastic_chance_values
                output["chance_encoder_embeddings"] = chance_encoder_embeddings

            # --- VALIDATION ---
            if shape_validator is not None:
                shape_validator.validate_predictions(output)

            return output

        # ----------------------------------------
        # PPO Logic
        # ----------------------------------------
        elif "policy_head" in self.components and "value_head" in self.components:
            x = initial_observation
            if "backbone" in self.components:
                x = self.components["backbone"](x)
            elif "feature_block" in self.components:
                x = self.components["feature_block"](x)

            policy_logits, _, _ = self.components["policy_head"](x)
            value_logits, _, _ = self.components["value_head"](x)

            output = {
                "values": value_logits.unsqueeze(1),
                "policies": policy_logits.unsqueeze(1),
            }

            # --- VALIDATION ---
            if shape_validator is not None:
                shape_validator.validate_predictions(output)

            return output

        # ----------------------------------------
        # Rainbow DQN Logic
        # ----------------------------------------
        elif "q_head" in self.components:
            x = self.components["feature_block"](initial_observation)
            Q_logits, _, _ = self.components["q_head"](x)
            q_vals = self.components["q_head"].representation.to_expected_value(
                Q_logits
            )

            output = {
                "q_values": q_vals.unsqueeze(1),
                "q_logits": Q_logits.unsqueeze(1),
            }

            # --- VALIDATION ---
            if shape_validator is not None:
                shape_validator.validate_predictions(output)

            return output

        # ----------------------------------------
        # Supervised/Imitation Logic
        # ----------------------------------------
        elif "policy_head" in self.components:
            x = initial_observation
            if "feature_block" in self.components:
                x = self.components["feature_block"](initial_observation)
            logits, _, _ = self.components["policy_head"](x)
            output = {"policies": logits.unsqueeze(1)}

            # --- VALIDATION ---
            if shape_validator is not None:
                shape_validator.validate_predictions(output)

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
        if "world_model" not in self.components or not self.stochastic:
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
