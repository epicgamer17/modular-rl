from operator import ge
import sys
from time import time

import torch
from torch.nn.utils import clip_grad_norm_

from agent_configs import PPOConfig
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

from utils import (
    normalize_policy,
    action_mask,
    get_legal_moves,
    update_linear_schedule,
)
from utils.utils import clip_low_prob_actions, normalize_policies, get_lr_scheduler
from modules.utils import get_optimal_device

sys.path.append("../")

import datetime
from modules.ppo import Network
from replay_buffers.base_replay_buffer import BasePPOReplayBuffer
from agents.agent import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: PPOConfig,
        name=datetime.datetime.now().timestamp(),
        device: torch.device = get_optimal_device(),
        from_checkpoint=False,
    ):
        super(PPOAgent, self).__init__(
            env, config, name, device=device, from_checkpoint=from_checkpoint
        )
        self.model = Network(
            config=config,
            output_size=self.num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
            discrete=self.discrete_action_space,  # COULD USE GAME CONFIG?
        ).to(self.device)

        if self.config.actor.optimizer == Adam:
            self.actor_optimizer: torch.optim.Optimizer = self.config.actor.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.actor.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.actor_optimizer: torch.optim.Optimizer = self.config.actor.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        if self.config.critic.optimizer == Adam:
            self.critic_optimizer: torch.optim.Optimizer = self.config.critic.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.critic.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.critic_optimizer: torch.optim.Optimizer = self.config.critic.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        
        self.actor_scheduler = get_lr_scheduler(self.actor_optimizer, self.config)
        self.critic_scheduler = get_lr_scheduler(self.critic_optimizer, self.config)

        # self.actor = ActorNetwork(
        #     input_shape=self.observation_dimensions,
        #     output_shape=self.num_actions,
        #     discrete=self.discrete_action_space,
        #     config=config,
        # )

        # self.critic = CriticNetwork(
        #     input_shape=self.observation_dimensions,
        #     config=config,
        # )

        self.replay_buffer = BasePPOReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.env.observation_space.dtype,
            max_size=self.config.replay_buffer_size,
            gamma=self.config.discount_factor,
            gae_lambda=self.config.gae_lambda,
        )

        self.stats = {
            "score": [],
            "actor_loss": [],
            "critic_loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
            "actor_loss": self.config.target_kl,
        }

    def checkpoint_optimizer_state(self, checkpoint):
        checkpoint["actor_optimizer"] = self.actor_optimizer.state_dict()
        checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()
        checkpoint["actor_scheduler"] = self.actor_scheduler.state_dict()
        checkpoint["critic_scheduler"] = self.critic_scheduler.state_dict()
        return checkpoint

    def load_optimizer_state(self, checkpoint):
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "actor_scheduler" in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler"])
        if "critic_scheduler" in checkpoint:
            self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler"])

    def predict(self, state, info: dict = None, mask_actions: bool = True):
        assert info is not None if mask_actions else True, "Need info to mask actions"
        state_input = self.preprocess(state)
        value = self.model.critic(inputs=state_input)
        if self.discrete_action_space:
            policy = self.model.actor(inputs=state_input)[0]
            if mask_actions:
                legal_moves = get_legal_moves(info)
                policy = action_mask(
                    policy, legal_moves, mask_value=0, device=self.device
                )
                policy = clip_low_prob_actions(policy, self.config.clip_low_prob)
                policy = normalize_policies(policy)
            distribution = torch.distributions.Categorical(probs=policy)
        else:
            mean, std = self.model.actor(inputs=state_input)
            distribution = torch.distributions.Normal(mean, std)
        return distribution, value

    def select_actions(self, predictions, *args, **kwargs):
        distribution = predictions[0]
        selected_action = distribution.sample()

        return selected_action

    def actor_learn(
        self,
        inputs,
        actions,
        log_probabilities,
        advantages,
        info: dict = None,
        mask_actions: bool = True,
    ):
        assert info is not None if mask_actions else True, "Need info to mask actions"
        # print("Training Actor")
        inputs = inputs.to(self.device)
        actions = actions.to(self.device)
        log_probabilities = log_probabilities.to(self.device)
        advantages = advantages.to(self.device)

        if self.discrete_action_space:
            probabilities = self.model.actor(inputs)
            if mask_actions:
                legal_moves = get_legal_moves(info)
                probabilities = action_mask(probabilities, legal_moves, mask_value=0)
                probabilities = normalize_policies(probabilities)
            distribution = torch.distributions.Categorical(probabilities)
        else:
            mean, std = self.model.actor(inputs)
            distribution = torch.distributions.Normal(mean, std)

        # tensor_actions = (
        #     torch.clone(actions).to(torch.float16).detach().requires_grad_(True)
        # )

        log_ratios = distribution.log_prob(actions) - log_probabilities

        probability_ratios = torch.exp(log_ratios)
        # min_advantages = tf.where(
        #     advantages > 0,
        #     (1 + self.clip_param) * advantages,
        #     (1 - self.clip_param) * advantages,
        # )

        clipped_probability_ratios = torch.clamp(
            probability_ratios,
            1 - self.config.clip_param,
            1 + self.config.clip_param,
        )

        # print((probability_ratios * advantages).shape)
        # print((clipped_probability_ratios * advantages).shape)

        actor_loss = torch.max(
            -probability_ratios * advantages, -clipped_probability_ratios * advantages
        ).mean()

        entropy_loss = distribution.entropy().mean()
        actor_loss = actor_loss - (self.config.entropy_coefficient * entropy_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.actor.clipnorm > 0:
            clip_grad_norm_(self.model.actor.parameters(), self.config.actor.clipnorm)

        self.actor_optimizer.step()
        with torch.no_grad():
            kl_divergence = torch.mean(
                log_probabilities - distribution.log_prob(actions)
            )
            kl_divergence = torch.sum(kl_divergence)
            print("Open AI Spinning Up KL Divergence", kl_divergence)
            approx_kl = ((probability_ratios - 1) - log_ratios).mean()
            print(
                "37 Implimentation Details KL Divergence",
                approx_kl,
            )

        return approx_kl.detach()

    def critic_learn(self, inputs, returns):
        inputs = inputs.to(self.device)
        returns = returns.to(self.device)

        critic_loss = (
            self.config.critic_coefficient * (returns - self.model.critic(inputs)) ** 2
        ).mean()

        print("critic loss", critic_loss)
        print(critic_loss.requires_grad)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.critic.clipnorm > 0:
            clip_grad_norm_(self.model.critic.parameters(), self.config.critic.clipnorm)

        self.critic_optimizer.step()
        return critic_loss.detach()

    def learn(self):
        samples = self.replay_buffer.sample()
        observations = samples["observations"]
        # actions = torch.from_numpy(samples["actions"])
        # log_probabilities = torch.from_numpy(samples["log_probabilities"])
        # advantages = torch.from_numpy(samples["advantages"])
        # returns = torch.from_numpy(samples["returns"])
        # infos = torch.from_numpy(samples["infos"])
        actions = samples["actions"].to(self.device)
        log_probabilities = samples["log_probabilities"].to(self.device)
        advantages = samples["advantages"].to(self.device)
        returns = samples["returns"].to(self.device)
        infos = samples["infos"].to(self.device)
        inputs = self.preprocess(observations)

        indices = torch.randperm(len(observations))
        minibatch_size = len(observations) // self.config.num_minibatches

        # actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.actor_optimizer,
        #     self.config.actor.learning_rate,
        #     0,
        #     total_iters=self.config.train_policy_iterations,
        # )

        for iteration in range(self.config.train_policy_iterations):
            # actor_scheduler.step()
            # print(actor_scheduler.get_last_lr())

            for start in range(0, len(observations), minibatch_size):
                end = start + minibatch_size
                batch_indices = indices[start:end]
                batch_observations = inputs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probabilities = log_probabilities[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_info = infos[batch_indices]
                kl_divergence = self.actor_learn(
                    batch_observations,
                    batch_actions,
                    batch_log_probabilities,
                    batch_advantages,
                    batch_info,
                )
                self.stats["actor_loss"].append(kl_divergence)
            if kl_divergence > 1.5 * self.config.target_kl:
                print("Early stopping at iteration {}".format(iteration))
                break
            # kl_divergence = self.train_actor(
            #     inputs, actions, log_probabilities, advantages, learning_rate
            # )
            # stat_actor_loss.append(kl_divergence)
            # if kl_divergence > 1.5 * self.target_kl:
            #     print("Early stopping at iteration {}".format(_))
            #     break
        # critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.critic_optimizer,
        #     self.config.critic.learning_rate,
        #     0,
        #     total_iters=self.config.train_value_iterations,
        # )
        for iteration in range(self.config.train_value_iterations):
            # critic_scheduler.step()
            # print(critic_scheduler.get_last_lr())

            for start in range(0, len(observations), minibatch_size):
                end = start + minibatch_size
                batch_indices = indices[start:end]
                batch_observations = inputs[batch_indices]
                batch_returns = returns[batch_indices]
                critic_loss = self.critic_learn(
                    batch_observations,
                    batch_returns,
                )
                self.stats["critic_loss"].append(critic_loss)
                # critic_loss = self.train_critic(inputs, returns, learning_rate)
                # stat_critic_loss.append(critic_loss)
                # stat_loss.append(critic_loss)

    def train(self):
        super().train()

        start_time = time() - self.training_time
        state, info = self.env.reset()

        while self.training_step < self.config.training_steps:
            with torch.no_grad():

                num_episodes = 0
                score = 0
                for timestep in range(self.config.steps_per_epoch):
                    predictions = self.predict(state, info)
                    action = self.select_actions(predictions).item()

                    next_state, reward, terminated, truncated, next_info = (
                        self.env.step(action)
                    )

                    distribution, value = predictions
                    log_probability = distribution.log_prob(torch.tensor(action))
                    value = value[0][0]

                    self.replay_buffer.store(
                        state, info, action, value, log_probability, reward
                    )

                    done = terminated or truncated
                    state = next_state
                    info = next_info
                    score += reward

                    if done or timestep == self.config.steps_per_epoch - 1:
                        last_value = (
                            0
                            if done
                            else self.model.critic(self.preprocess(next_state))
                        )
                        self.replay_buffer.finish_trajectory(last_value)
                        num_episodes += 1
                        state, info = self.env.reset()
                        score_dict = {"score": score}
                        self.stats["score"].append(score_dict)
                        score = 0

            self.learn()
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            # self.old_actor.set_weights(self.actor.get_weights())
            if self.training_step % self.checkpoint_interval == 0:
                self.training_time = time() - start_time
                self.total_environment_steps += self.config.steps_per_epoch
                self.save_checkpoint()
            self.training_step += 1

        self.training_time = time() - start_time
        self.total_environment_steps = (
            self.config.training_steps * self.config.steps_per_epoch
        )
        self.save_checkpoint()
        self.env.close()
