import math
from time import time
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
import numpy as np
from agent_configs.dqn.rainbow_config import RainbowConfig
from agents.catan_player_wrapper import ACTIONS_ARRAY
from replay_buffers.buffer_factories import create_dqn_buffer
from replay_buffers.processors import NStepInputProcessor, StandardOutputProcessor
from replay_buffers.samplers import PrioritizedSampler
from replay_buffers.writers import CircularWriter
from utils.utils import (
    get_legal_moves,
    current_timestamp,
    action_mask,
    epsilon_greedy_policy,
    update_inverse_sqrt_schedule,
    update_linear_schedule,
    get_lr_scheduler,
)

from losses.basic_losses import (
    CategoricalCrossentropyLoss,
    HuberLoss,
    KLDivergenceLoss,
    MSELoss,
)
from losses.losses import LossPipeline, StandardDQNLoss, C51Loss

from replay_buffers.utils import update_per_beta
from modules.utils import get_optimal_device

import sys

sys.path.append("../../")

from agents.agent import BaseAgent
from modules.rainbow_dqn import RainbowNetwork
from stats.stats import PlotType, StatTracker


class RainbowAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: RainbowConfig,
        name=f"rainbow_{current_timestamp():.1f}",
        device: torch.device = get_optimal_device(),
        from_checkpoint=False,
    ):
        super(RainbowAgent, self).__init__(env, config, name, device=device)
        self.model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
        )
        self.target_model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
        )

        if not self.config.kernel_initializer == None:
            self.model.initialize(self.config.kernel_initializer)

        self.model.to(device)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        loss_modules = []

        # Logic to decide which "Main" loss to use
        if self.config.atom_size > 1:
            # If config asks for distributional RL, use C51
            loss_modules.append(C51Loss(self.config, self.device))
        else:
            # Otherwise, use Standard DQN
            loss_modules.append(StandardDQNLoss(self.config, self.device))

        # Finalize the pipeline
        self.loss_pipeline = LossPipeline(loss_modules)

        if self.config.optimizer == Adam:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.config)

        self.replay_buffer = create_dqn_buffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            num_actions=self.num_actions,
            batch_size=self.config.minibatch_size,
            observation_dtype=self.observation_dtype,
            config=self.config,
        )

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?)
        # self.v_min = self.config.v_min
        # self.v_max = self.config.v_max

        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=device,
        ).to(device)
        """row vector Tensor(atom_size)
        """

        self.eg_epsilon = self.config.eg_epsilon

        self.stats = StatTracker(
            model_name=self.model_name,
            stat_keys=[
                "score",
                "loss",
                "test_score",
            ],
            target_values={
                "score": (
                    self.env.spec.reward_threshold
                    if hasattr(self.env, "spec") and self.env.spec.reward_threshold
                    else None
                ),
                "test_score": (
                    self.env.spec.reward_threshold
                    if hasattr(self.env, "spec") and self.env.spec.reward_threshold
                    else None
                ),
            },
            use_tensor_dicts={
                "test_score": ["score", "max_score", "min_score"],
            },
        )
        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
            PlotType.BEST_FIT_LINE,
            rolling_window=100,
        )
        self.stats.add_plot_types(
            "test_score",
            PlotType.BEST_FIT_LINE,
            PlotType.ROLLING_AVG,
            rolling_window=100,
        )
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)

    def checkpoint_model_weights(self, checkpoint):
        checkpoint = super().checkpoint_model_weights(checkpoint)
        checkpoint["target_model"] = self.target_model.state_dict()
        return checkpoint

    def load_model_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.target_model.eval()

    def predict(self, states, *args, **kwargs) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.model(state_input)
        return q_distribution

    def predict_target(self, states) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.target_model(state_input)
        return q_distribution

    def select_actions(
        self,
        distribution,
        info: dict = None,
    ):

        if self.config.atom_size > 1:
            q_values = distribution * self.support
            q_values = q_values.sum(2, keepdim=False)
        else:
            q_values = distribution
        if "legal_moves" in info:
            legal_moves = get_legal_moves(info)
            q_values = action_mask(
                q_values, legal_moves, mask_value=-float("inf"), device=self.device
            )
        # q_values with argmax ties
        # selected_actions = torch.stack(
        #     [
        #         torch.tensor(np.random.choice(np.where(x.cpu() == x.cpu().max())[0]))
        #         for x in q_values
        #     ]
        # )
        selected_actions = q_values.argmax(1, keepdim=False)
        return selected_actions

    def learn(self) -> np.ndarray:
        losses = np.zeros(self.config.training_iterations)

        for i in range(self.config.training_iterations):
            # 1. Get Samples (Initialize Context)
            # The context dict starts with just the raw batch data
            context = self.replay_buffer.sample()

            # 2. Run Pipeline
            # This handles ensure_predictions -> ensure_targets -> compute_loss
            # It returns the sum of all losses and the primary elementwise loss for PER
            loss, elementwise_loss = self.loss_pipeline.run(self, context)

            # 3. Optimization
            self.optimizer.zero_grad()
            loss.backward()

            if self.config.clipnorm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clipnorm
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            # 4. Update Priorities (PER)
            # elementwise_loss is detached inside the pipeline logic if needed,
            # but usually safe to detach again here.
            self.update_replay_priorities(
                indices=context["indices"],
                priorities=elementwise_loss.detach(),
                ids=None,
            )

            # 5. Housekeeping
            losses[i] = loss.detach().item()
            self.model.reset_noise()
            self.target_model.reset_noise()

        return losses

    def fill_replay_buffer(self):
        print("replay buffer size:", self.replay_buffer.size)
        with torch.no_grad():
            state, info = self.env.reset()
            target_size = self.config.min_replay_buffer_size
            while self.replay_buffer.size < target_size:
                if (self.replay_buffer.size % (math.ceil(target_size / 100))) == 0:
                    print(
                        f"filling replay buffer: {self.replay_buffer.size} / ({target_size})"
                    )
                # dist = self.predict(state)
                # action = self.select_actions(dist).item()
                if "legal_moves" in info:
                    action = np.random.choice(info["legal_moves"])
                else:
                    action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, next_info = self.env.step(
                    action
                )
                done = terminated or truncated
                # print(state)
                self.replay_buffer.store(
                    observations=state,
                    actions=action,
                    rewards=reward,
                    next_observations=next_state,
                    next_infos=next_info,
                    dones=done,
                )
                # print(self.replay_buffer.observation_buffer[0])
                state = next_state
                info = next_info
                if done:
                    state, info = self.env.reset()
                # gc.collect()

    def update_eg_epsilon(self, training_step):
        if self.config.eg_epsilon_decay_type == "linear":
            self.eg_epsilon = update_linear_schedule(
                self.config.eg_epsilon_final,
                self.config.eg_epsilon_final_step,
                self.config.eg_epsilon,
                training_step,
            )
        elif self.config.eg_epsilon_decay_type == "inverse_sqrt":
            self.eg_epsilon = update_inverse_sqrt_schedule(
                self.config.eg_epsilon,
                training_step,
            )
        else:
            raise ValueError(
                "Invalid epsilon decay type: {}".format(
                    self.config.eg_epsilon_decay_type
                )
            )

    def train(self):
        start_time = time() - self.stats.get_time_elapsed()
        score = 0
        self.fill_replay_buffer()

        state, info = self.env.reset()

        while self.training_step < self.config.training_steps:
            with torch.no_grad():
                for _ in range(self.config.replay_interval):
                    values = self.predict(state)
                    action = epsilon_greedy_policy(
                        values,
                        info,
                        self.eg_epsilon,
                        wrapper=lambda values, info: self.select_actions(
                            values, info
                        ).item(),
                    )

                    next_state, reward, terminated, truncated, next_info = (
                        self.env.step(action)
                    )
                    done = terminated or truncated
                    # print("State", state)
                    self.replay_buffer.store(
                        observations=state,
                        actions=action,
                        rewards=reward,
                        next_observations=next_state,
                        next_infos=next_info,
                        dones=done,
                    )
                    state = next_state
                    info = next_info
                    score += reward
                    if done:
                        state, info = self.env.reset()
                        self.stats.append("score", score)
                        score = 0
            self.replay_buffer.set_beta(
                # TODO: MOVE THIS LOGIC INTO SAMPLER
                update_per_beta(
                    self.replay_buffer.beta,
                    self.config.per_beta_final,
                    self.config.training_steps,
                    self.config.per_beta,
                )
            )

            self.update_eg_epsilon(self.training_step + 1)
            for minibatch in range(self.config.num_minibatches):
                if len(self.replay_buffer) < self.config.min_replay_buffer_size:
                    break
                losses = self.learn()
                loss_mean = losses.mean()
                # could do things other than taking the mean here
                self.stats.append("loss", loss_mean)

            if self.training_step % self.config.transfer_interval == 0:
                self.update_target_model()

            if self.training_step % self.test_interval == 0:
                self.run_tests(self.stats)

            if self.training_step % self.checkpoint_interval == 0:
                self.stats.set_time_elapsed(time() - start_time)
                self.stats.increment_steps(
                    self.training_step * self.config.replay_interval
                )
                self.save_checkpoint(save_weights=self.config.save_intermediate_weights)
            # gc.collect()
            self.training_step += 1

        self.stats.set_time_elapsed(time() - start_time)
        self.stats.increment_steps(self.training_step * self.config.replay_interval)
        self.save_checkpoint(save_weights=True)
        # self.env.close()
