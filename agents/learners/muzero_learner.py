from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
import time
from torch.nn.utils import clip_grad_norm_
from replay_buffers.buffer_factories import create_muzero_buffer
from losses.losses import create_muzero_loss_pipeline
from modules.utils import get_lr_scheduler
from replay_buffers.utils import update_per_beta
from modules.world_models.inference_output import LearningOutput


class MuZeroLearner:
    """
    MuZeroLearner handles the training logic, including buffer management,
    optimizer stepping, and loss computation.
    """

    def __init__(
        self,
        config,
        model,
        device,
        num_actions,
        observation_dimensions,
        observation_dtype,
        # policy, # REMOVED
    ):
        self.config = config
        self.model = model
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        # self.policy = policy # REMOVED

        self.training_step = 0

        # 1. Initialize Replay Buffer
        self.replay_buffer = create_muzero_buffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            num_actions=self.num_actions,
            num_players=self.config.game.num_players,
            unroll_steps=self.config.unroll_steps,
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
            batch_size=self.config.minibatch_size,
            observation_dtype=observation_dtype,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            epsilon=self.config.per_epsilon,
            use_batch_weights=self.config.per_use_batch_weights,
            use_initial_max_priority=self.config.per_use_initial_max_priority,
            lstm_horizon_len=self.config.lstm_horizon_len,
            value_prefix=self.config.value_prefix,
            tau=self.config.reanalyze_tau,
            multi_process=self.config.multi_process,
        )

        # 2. Initialize Optimizer
        if self.config.optimizer == Adam:
            self.optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            self.optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        # 3. Initialize Scheduler
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.config)

        # 4. Initialize Loss Pipeline
        self.loss_pipeline = create_muzero_loss_pipeline(
            config=self.config,
            device=self.device,
            model=self.model,
        )

    def _preprocess_observation(self, states: Any) -> torch.Tensor:
        """
        Converts states to torch tensors on the correct device.
        Adds batch dimension if input is a single observation.
        """
        if torch.is_tensor(states):
            if states.device == self.device and states.dtype == torch.float32:
                prepared_state = states
            else:
                prepared_state = states.to(self.device, dtype=torch.float32)
        else:
            np_states = np.array(states, copy=False)
            prepared_state = torch.tensor(
                np_states, dtype=torch.float32, device=self.device
            )

        if prepared_state.ndim == 0:
            prepared_state = prepared_state.unsqueeze(0)

        # states might be a single observation without batch dim
        # But for Learner, we expect batch of samples.
        # If the input shape matches single observation, unsqueeze?
        # Replay buffer samples are usually batched: [B, ...]
        # So we probably don't need the strict unsqueeze check for single item unless 'states' is a single step from pipeline?
        # The original SearchPolicy.preprocess handled both.
        # Here we usually get [B, *obs_shape].

        return prepared_state

    def step(self, stats=None):
        """
        Performs a single training step.
        Returns a dictionary of loss statistics or None if buffer is too small.
        """
        if self.replay_buffer.size < self.config.min_replay_buffer_size:
            return None

        # Sample from buffer
        samples = self.replay_buffer.sample()

        # Internal learn logic
        loss_results = self._learn_step(samples, stats)

        # Update priorities
        if loss_results.get("priorities") is not None:
            priorities = loss_results["priorities"]
            # Support for both tensor and numpy priorities
            if isinstance(priorities, torch.Tensor):
                priorities = priorities.cpu().numpy()
            self.replay_buffer.update_priorities(
                samples["indices"], priorities, ids=samples.get("ids")
            )

        # Update PER beta
        self.replay_buffer.set_beta(
            update_per_beta(
                self.replay_buffer.beta,
                self.config.per_beta_final,
                self.config.training_steps,
                self.config.per_beta,
            )
        )

        self.training_step += 1
        return loss_results["stats"]

    def _learn_step(self, samples, stats=None):
        """Internal logic for a single backprop step."""
        start_time = time.time()

        # 1. Preprocess observations
        inputs = self._preprocess_observation(samples["observations"])
        samples["observations"] = inputs

        # 2. Extract Data and Masks
        has_valid_obs_mask = samples["obs_mask"].to(self.device).bool()
        has_valid_action_mask = samples["action_mask"].to(self.device).bool()
        target_observations = samples["unroll_observations"].to(self.device)

        weights = samples["weights"].to(self.device).float()

        # 3. Calculate Gradient Scales
        unroll_steps = self.config.unroll_steps
        gradient_scales = [1.0] + [1.0 / unroll_steps] * unroll_steps
        gradient_scales_tensor = torch.tensor(
            gradient_scales, device=self.device
        ).reshape(1, -1)

        # 3. Run Unroll (Learner Inference)
        network_output = self.model.learner_inference(samples)

        # 4. Extract Targets
        target_values = samples["values"].to(self.device)
        target_rewards = samples["rewards"].to(self.device)
        target_policies = samples["policies"].to(self.device)
        target_to_plays = samples["to_plays"].to(self.device)
        target_chance_codes = samples["chance_codes"].to(
            self.device
        )  # Keep this for stochastic stats if needed
        actions = samples["actions"].to(
            self.device
        )  # Keep this for latent viz if needed

        # 5. Build Tensors for LossPipeline
        predictions_tensor = {
            "values": network_output.values,
            "policies": network_output.policies,
            "rewards": network_output.rewards,
            "latent_states": network_output.latents,
        }

        if network_output.to_plays is not None:
            predictions_tensor["to_plays"] = network_output.to_plays

        if network_output.latents_afterstates is not None:
            predictions_tensor["latent_afterstates"] = (
                network_output.latents_afterstates
            )

        if network_output.chance_values is not None:
            predictions_tensor["chance_values"] = network_output.chance_values

        if network_output.chance_logits is not None:
            predictions_tensor["latent_code_probabilities"] = (
                network_output.chance_logits
            )

        # Stochastic MuZero extra fields (if present in UnrollOutput extras or main fields)
        if self.config.stochastic and network_output.extras:
            if "encoder_softmaxes" in network_output.extras:
                predictions_tensor["encoder_softmaxes"] = network_output.extras[
                    "encoder_softmaxes"
                ]

        targets_tensor = {
            "values": target_values,
            "rewards": target_rewards,
            "policies": target_policies,
            "to_plays": target_to_plays,
            "latent_states": network_output.target_latents,
        }

        if self.config.stochastic:
            # Chance values target: v_{t+1} (or similar depending on implementation)
            # For afterstate s_t^a, we predict v(s_{t+1}) or something similar.
            # Usually it's v(s_{t+1}) but sign-flipped if multi-player.
            targets_tensor["chance_values"] = target_values[:, 1:]
            if network_output.extras and "encoder_onehots" in network_output.extras:
                targets_tensor["encoder_onehots"] = network_output.extras[
                    "encoder_onehots"
                ]

        # 7. Context for vectorized masking
        context = {
            "has_valid_obs_mask": has_valid_obs_mask,
            "has_valid_action_mask": has_valid_action_mask,
            "target_observations": target_observations,
        }

        # 8. Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        loss_mean, loss_dict, priorities = self.loss_pipeline.run(
            predictions_tensor=predictions_tensor,
            targets_tensor=targets_tensor,
            context=context,
            weights=weights,
            gradient_scales=gradient_scales_tensor,
            config=self.config,
            device=self.device,
        )

        if stats is not None:
            if self.config.stochastic:
                self._track_stochastic_stats(
                    targets_tensor["encoder_onehots"],
                    predictions_tensor["latent_code_probabilities"],
                    stats,
                )

            if self.training_step % self.config.latent_viz_interval == 0:
                self._track_latent_visualization(
                    predictions_tensor["latent_states"], actions, stats
                )

        loss_mean.backward()

        if self.config.clipnorm > 0:
            clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

        self.optimizer.step()
        self.lr_scheduler.step()

        if stats is not None:
            duration = time.time() - start_time
            if duration > 0:
                fps = self.config.minibatch_size / duration
                stats.append("learner_fps", fps)

        if self.device == "mps" and self.training_step % 100 == 0:
            torch.mps.empty_cache()

        return {
            "stats": self._prepare_stats(loss_dict, loss_mean.item()),
            "priorities": priorities,
            "predictions": {k: v.detach() for k, v in predictions_tensor.items()},
            "targets": {k: v.detach() for k, v in targets_tensor.items()},
            "masks": (has_valid_action_mask, has_valid_obs_mask),
            "actions": actions.detach(),
        }

    def _track_latent_visualization(self, latent_states, actions, stats):
        """Track latent space representations categorized by action."""
        if stats is None:
            return
        s0 = latent_states[:, 0].detach().cpu()
        a0 = actions[:, 0].detach().cpu()
        stats.add_latent_visualization(
            "latent_root", s0, labels=a0, method=self.config.latent_viz_method
        )

    def _track_stochastic_stats(
        self, encoder_onehots_tensor, latent_code_probs_tensor, stats
    ):
        """Track statistics for stochastic MuZero."""
        if stats is None:
            return
        if latent_code_probs_tensor.ndim == 3:
            prob_sums = latent_code_probs_tensor.sum(dim=-1)
            mask = prob_sums > 0.001
        else:
            mask = torch.ones(
                latent_code_probs_tensor.shape[:-1],
                dtype=torch.bool,
                device=self.device,
            )

        codes = encoder_onehots_tensor.argmax(dim=-1)
        if mask.shape == codes.shape:
            valid_codes = codes[mask]
        else:
            valid_codes = codes.flatten()

        unique_codes_all = torch.unique(valid_codes)
        num_unique_all_int = int(unique_codes_all.numel())
        stats.append("num_codes", num_unique_all_int)

        latent_node_probs = latent_code_probs_tensor
        if latent_node_probs.ndim == 3:
            valid_probs = latent_node_probs[mask]
            if valid_probs.shape[0] > 0:
                mean_probs = valid_probs.mean(dim=0)
            else:
                mean_probs = torch.zeros(
                    latent_node_probs.shape[-1], device=latent_node_probs.device
                )
        else:
            mean_probs = latent_node_probs.mean(dim=0)

        stats.append("chance_probs", mean_probs.detach().cpu())

        probs = latent_code_probs_tensor
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if mask is not None:
            if entropy.shape == mask.shape:
                entropy = entropy[mask]
            else:
                entropy = entropy.flatten()

        mean_entropy = entropy.mean().item() if entropy.numel() > 0 else 0.0
        stats.append("chance_entropy", mean_entropy)

    def _prepare_stats(self, loss_dict, total_loss):
        def get_val(key):
            val = loss_dict.get(key, 0.0)
            return val.item() if isinstance(val, torch.Tensor) else val

        return {
            "value_loss": get_val("ValueLoss"),
            "policy_loss": get_val("PolicyLoss"),
            "reward_loss": get_val("RewardLoss"),
            "to_play_loss": get_val("ToPlayLoss"),
            "cons_loss": get_val("ConsistencyLoss"),
            "q_loss": get_val("ChanceQLoss"),
            "sigma_loss": get_val("SigmaLoss"),
            "vqvae_commitment_cost": get_val("VQVAECommitmentLoss"),
            "loss": total_loss,
        }
