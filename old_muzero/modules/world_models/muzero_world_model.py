from typing import Callable, List, Optional, Tuple, Dict, Any, Union

from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F

from old_muzero.configs.agents.muzero import MuZeroConfig
from old_muzero.modules.backbones.factory import BackboneFactory
from old_muzero.modules.heads.factory import HeadFactory
from old_muzero.modules.heads.to_play import ToPlayHead
from old_muzero.modules.heads.reward import RewardHead
from old_muzero.agents.learner.losses.representations import get_representation
from old_muzero.modules.utils import scale_gradient, kernel_initializer_wrapper
from old_muzero.modules.world_models.inference_output import (
    MuZeroNetworkState,
    WorldModelOutput,
    PhysicsOutput,
)
from old_muzero.modules.world_models.world_model import WorldModelInterface

from old_muzero.modules.world_models.components.representation import Representation
from old_muzero.modules.world_models.components.dynamics import Dynamics, AfterstateDynamics
from old_muzero.modules.world_models.components.chance_encoder import ChanceEncoder

from old_muzero.modules.world_models.world_model import WorldModelOutput


class MuzeroWorldModel(WorldModelInterface, nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        observation_dimensions: Tuple[int, ...],
        num_actions: int,
    ):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        # 1. Representation Network
        self.representation = Representation(config, observation_dimensions)
        self.num_chance = config.num_chance

        hidden_state_shape = self.representation.output_shape

        # --- 3. Dynamics and Prediction Networks ---
        if self.config.stochastic:
            self.afterstate_dynamics = AfterstateDynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
            )

            # Stochastic Dynamics(Afterstate, Code) -> Next State
            self.dynamics = Dynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_chance,
                action_embedding_dim=self.config.action_embedding_dim,
            )

            # Shared Backbone for Afterstate Features (Sigma and Value Heads)
            self.shared_backbone = BackboneFactory.create(
                self.config.prediction_backbone, self.representation.output_shape
            )

            # Sigma Head (Owned by WorldModel - fundamental environment rules)
            self.sigma_head = HeadFactory.create(
                self.config.chance_probability_head,
                self.config.arch,
                input_shape=self.shared_backbone.output_shape,
                num_chance_codes=self.num_chance,
            )

            # Encoder (Moves from AgentNetwork to WorldModel)
            encoder_input_shape = list(observation_dimensions)
            encoder_input_shape[0] *= 2  # Double channels for stacked obs
            self.encoder = ChanceEncoder(
                config,
                tuple(encoder_input_shape),
                num_codes=self.num_chance,
            )

        else:
            self.dynamics = Dynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
            )

        # 3. Physics Heads (Owned by WorldModel now)
        # Reward Head
        r_rep = get_representation(config.reward_head.output_strategy)
        self.reward_head = HeadFactory.create(
            config.reward_head,
            config.arch,
            input_shape=self.dynamics.output_shape,
            representation=r_rep,
        )

        # To-Play Head
        tp_rep = get_representation(config.to_play_head.output_strategy)
        self.to_play_head = ToPlayHead(
            arch_config=config.arch,
            input_shape=self.dynamics.output_shape,
            num_players=config.game.num_players,
            neck_config=config.to_play_head.neck,
            representation=tp_rep,
        )

        # Dynamics output must match Representation output shape
        assert (
            self.dynamics.output_shape == self.representation.output_shape
        ), f"{self.dynamics.output_shape} = {self.representation.output_shape}"

    @property
    def device(self) -> torch.device:
        return (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

    def initialize(
        self, initializer: Optional[Union[Callable[[Tensor], None], str]] = None
    ) -> None:
        """
        Unified initialization for the World Model.
        Recursively applies the initializer function to all applicable layers (Conv, Linear).
        """
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

    def initial_inference(self, observation: Tensor) -> WorldModelOutput:
        # Ensure observation is a tensor
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            )

        # Ensure observation has batch dim
        if observation.dim() == len(self.representation.input_shape):
            observation = observation.unsqueeze(0)

        hidden_state = self.representation(observation.float())
        return WorldModelOutput(features=hidden_state)

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        recurrent_state: Any = None,
    ) -> WorldModelOutput:
        if not self.config.stochastic:
            action = action.view(-1).to(hidden_state.device)
            # one-hot the action -> (B, num_actions)
            action = (
                F.one_hot(action.long(), num_classes=self.num_actions)
                .float()
                .to(hidden_state.device)
            )

        next_hidden_state = self.dynamics(hidden_state, action)

        reward_logits, new_state, instant_reward = self.reward_head(
            next_hidden_state, state=recurrent_state
        )
        # For return, we might want to return the full opaque state as "head_state" field
        # or separate fields? properties of WorldModelOutput are flexible?
        # WorldModelOutput.head_state is Any.
        outcome_state = new_state if isinstance(recurrent_state, dict) else new_state

        # to_play_head returns (logits, state, player_idx).
        # Recurrent inference exposes logits for the learner, player_idx for the actor/MCTS.
        to_play_logits, _, to_play = self.to_play_head(next_hidden_state)

        return WorldModelOutput(
            features=next_hidden_state,
            reward=reward_logits,
            to_play=to_play,  # actor-facing: scalar player index (B,)
            to_play_logits=to_play_logits,  # learner-facing: raw logits (B, num_players)
            head_state=outcome_state,
            instant_reward=instant_reward,
        )

    def afterstate_recurrent_inference(
        self,
        network_state: Dict[str, Any],
        action: Tensor,
    ) -> WorldModelOutput:
        """
        Computes the afterstate from a deterministic action (stochastic MuZero).

        This method is called from two places:
        - ``unroll_physics`` (internal): passes a raw latent Tensor directly.
        - ``MuZeroNetwork.afterstate_inference`` (MCTS external): passes the
          opaque ``{"dynamics": Tensor, "wm_memory": ...}`` dict.

        Both cases are handled gracefully via ``MuZeroNetworkState``.

        Args:
            network_state: A ``MuZeroNetworkState`` — either from
                ``MuZeroNetwork.afterstate_inference`` (MCTS) or wrapped by
                ``unroll_physics`` (learner). Must never be a raw Tensor.
            action: Action taken at this step.

        Returns:
            WorldModelOutput with afterstate features and chance logits.
        """
        assert isinstance(network_state, MuZeroNetworkState), (
            f"network_state must be a MuZeroNetworkState, got {type(network_state)}. "
            "Wrap raw latent tensors with MuZeroNetworkState(dynamics=latent) before calling."
        )
        latent_state = network_state.dynamics

        # 1. Transition to Afterstate
        action = action.view(-1).to(latent_state.device)
        # one-hot the action -> (B, num_actions)
        action = (
            F.one_hot(action.long(), num_classes=self.num_actions)
            .float()
            .to(latent_state.device)
        )

        afterstate_latent = self.afterstate_dynamics(latent_state, action)

        # 2. Extract Shared Features (Backbone)
        shared_features = self.shared_backbone(afterstate_latent)

        # 3. Predict Chance Probabilities (Logits)
        chance_logits, _, _ = self.sigma_head(shared_features)

        return WorldModelOutput(
            afterstate_features=afterstate_latent,  # Raw dynamics output
            features=shared_features,  # Shared processed features for Agent
            chance=chance_logits,  # Environment chance rules
        )

    def unroll_physics(
        self,
        initial_latent_state: Tensor,
        actions: Tensor,
        encoder_inputs: Tensor,
        true_chance_codes: Tensor,
        head_state: Any,
        target_observations: Optional[Tensor] = None,
    ) -> PhysicsOutput:
        """
        Unrolls the dynamics for K steps given actions.
        Returns a PhysicsOutput containing STACKED tensors for each step.
        """
        batch_size = actions.shape[0]
        # Use the actual number of provided actions rather than config.unroll_steps
        # so callers (e.g. tests) may pass shorter sequences without errors.
        unroll_steps = actions.shape[
            1
        ]  # TODO: this really should be config.unroll_steps
        device = initial_latent_state.device

        # --- 2. Prepare Storage ---
        latent_states = []
        rewards = []
        to_plays = []
        latent_afterstates = []  # For stochastic MuZero
        latent_code_probabilities = []  # For stochastic MuZero
        afterstate_backbone_features = []  # For stochastic MuZero Value Head

        # Current states
        hidden_states = initial_latent_state
        # Initialize opaque reward state
        # We assume start of unroll is step 0 for horizon purposes
        current_head_state = head_state

        # Add initial states to lists
        latent_states.append(hidden_states)

        # Predict initial player — return logits so ToPlayLoss receives pre-softmax
        # values (cross_entropy expects logits, not probabilities).
        # to_play_head returns (logits, state, player_idx); store logits for the learner.
        initial_to_play_logits, _, _player_idx = self.to_play_head(hidden_states)
        to_plays.append(initial_to_play_logits)

        # Stochastic MuZero specific storage
        if self.config.stochastic:
            latent_afterstates = []
            latent_code_probabilities = []
            chance_encoder_embeddings = []
            chance_encoder_onehots = []
            afterstate_backbone_features = []
        else:
            latent_afterstates = []
            latent_code_probabilities = []
            afterstate_backbone_features = []
            chance_encoder_embeddings = []
            chance_encoder_onehots = []

        # Rewards: MuZero unrolls usually return T rewards for T actions.
        # PhysicsOutput says rewards: [B, T, ...]
        rewards = []

        # --- 4. Unroll Loop ---
        for k in range(unroll_steps):
            actions_k = actions[:, k]

            if self.config.stochastic:
                # 1. Afterstate Inference — wrap latent in MuZeroNetworkState so the
                # method's strict type check passes (it must be a structured object,
                # never a raw Tensor).
                afterstate_out = self.afterstate_recurrent_inference(
                    MuZeroNetworkState(dynamics=hidden_states), actions_k
                )
                afterstates = afterstate_out.afterstate_features
                shared_features = afterstate_out.features
                chance_logits_k = afterstate_out.chance

                # 3. Encoder Inference
                chance_encoder_embedding_k, chance_encoder_onehot_k = self.encoder(encoder_inputs[:, k])

                if self.config.use_true_chance_codes:
                    codes_k = F.one_hot(
                        true_chance_codes[:, k + 1].squeeze(-1).long(),
                        self.config.num_chance,
                    )
                    chance_encoder_onehot_k = codes_k.float()

                latent_afterstates.append(afterstates)
                latent_code_probabilities.append(chance_logits_k)  # Logits
                afterstate_backbone_features.append(shared_features)

                # Store encoder outputs
                chance_encoder_onehots.append(chance_encoder_onehot_k)
                chance_encoder_embeddings.append(chance_encoder_embedding_k)

                # 4. Dynamics Inference (using chance code as action)
                next_hidden_state = self.dynamics(afterstates, chance_encoder_onehot_k)

                # Heads
                reward_logits, new_state, instant_reward = self.reward_head(
                    next_hidden_state, state=current_head_state
                )
                current_head_state = new_state

                # to_play_head returns (logits, state, player_idx).
                to_play_logits, _, to_play = self.to_play_head(next_hidden_state)

                rewards_k = reward_logits  # Changed
                hidden_states = next_hidden_state
                # Store logits for the learner (ToPlayLoss expects pre-softmax values).
                to_plays_k = to_play_logits

                # Update LSTM states (for potential external usage, though unroll mainly uses dict now)
                # But we don't strictly need these unpacking if we pass current_reward_state next time

            else:
                wm_output = self.recurrent_inference(
                    hidden_states,
                    actions_k,
                    current_head_state,
                )
                rewards_k = wm_output.reward
                hidden_states = wm_output.features
                # Use logits for the learner (cross_entropy expects pre-softmax values).
                to_plays_k = wm_output.to_play_logits

                # wm_output.head_state is now the opaque dict
                current_head_state = wm_output.head_state
                instant_rewards_k = wm_output.instant_reward  # New

            latent_states.append(hidden_states)
            rewards.append(rewards_k)
            to_plays.append(to_plays_k)

            # Scale the gradient of the hidden state (applies to the whole batch)
            hidden_states = scale_gradient(hidden_states, 0.5)

            # Horizon logic is now handled inside ValuePrefixRewardHead via step_count

        # --- 5. No Padding needed for K steps alignment ---

        # --- 6. Stack and Return PhysicsOutput ---
        # Stack everything here to avoid returning lists of tensors
        stacked_latents = torch.stack(latent_states, dim=1)

        # Rewards: size K (1...K)
        stacked_rewards = torch.stack(rewards, dim=1) if rewards else torch.empty(0)

        stacked_to_plays = torch.stack(to_plays, dim=1) if to_plays else torch.empty(0)

        stacked_afterstates = None
        stacked_chance_logits = None
        stacked_backbone = None
        stacked_chance_encoder_embeddings = None
        stacked_chance_encoder_onehots = None

        if self.config.stochastic:
            stacked_afterstates = torch.stack(latent_afterstates, dim=1)
            stacked_chance_logits = torch.stack(latent_code_probabilities, dim=1)
            stacked_backbone = torch.stack(afterstate_backbone_features, dim=1)
            stacked_chance_encoder_embeddings = torch.stack(chance_encoder_embeddings, dim=1)
            stacked_chance_encoder_onehots = torch.stack(chance_encoder_onehots, dim=1)

        # 7. Compute target latents for consistency loss if requested
        stacked_target_latents = None
        if target_observations is not None and self.config.consistency_loss_factor > 0:
            B_target, T_plus_1_target = target_observations.shape[:2]
            flat_target_obs = target_observations.reshape(
                B_target * T_plus_1_target, *target_observations.shape[2:]
            )
            # Encode target observations
            with torch.no_grad():
                target_latents = self.representation(flat_target_obs.float())
                stacked_target_latents = target_latents.view(
                    B_target, T_plus_1_target, *target_latents.shape[1:]
                )

        return PhysicsOutput(
            latents=stacked_latents,
            rewards=stacked_rewards,
            to_plays=stacked_to_plays,
            latents_afterstates=stacked_afterstates,
            chance_logits=stacked_chance_logits,
            afterstate_backbone_features=stacked_backbone,
            chance_encoder_embeddings=stacked_chance_encoder_embeddings,
            chance_encoder_onehots=stacked_chance_encoder_onehots,
            target_latents=stacked_target_latents,
        )

    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            "representation_network": self.representation,
            "dynamics_network": self.dynamics,
            "afterstate_dynamics_network": getattr(self, "afterstate_dynamics", None),
        }
