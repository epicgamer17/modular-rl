# baseline for NFSP, should work for any model type (paper uses DQN)
# Initialize game Γ and execute an agent via RUNAGENT for each player in the game
# function RUNAGENT(Γ)
# Initialize replay memories MRL (circular buffer) and MSL (reservoir)
# Initialize average-policy network Π(s, a | θΠ) with random parameters θΠ
# Initialize action-value network Q(s, a | θQ) with random parameters θQ
# Initialize target network parameters θQ0 ← θQ
# Initialize anticipatory parameter η
# for each episode do
# Set policy σ ← {
# epsilon-greedy (Q), with probability η
# Π, with probability 1 − η
# }
# Observe initial information state s1 and reward r1
# for t = 1, T do
# Sample action at from policy σ
# Execute action at in game and observe reward rt+1 and next information state st+1
# Store transition (st, at, rt+1, st+1) in reinforcement learning memory MRL
# if agent follows best response policy σ = epsilon-greedy (Q) then
# Store behaviour tuple (st, at) in supervised learning memory MSL
# end if
# Update θΠ with stochastic gradient descent on loss
# L(θΠ) = E(s,a) from MSL [log Π(s, a | θΠ)]
# Update θQ with stochastic gradient descent on loss
# LθQ = E(s,a,r,s0) from MRL [r + maxa0 Q(s0, a0| θQ0) − Q(s, a | θQ)^2]
# Periodically update target network parameters θ   Q0 ← θQ
# end for
# end for
# end function

import gc
import os
from pathlib import Path
import pickle
import sys
from time import time
from tqdm import tqdm
import numpy as np
from agent_configs import NFSPDQNConfig
import copy

import torch
from stats.stats import PlotType, StatTracker
from utils import (
    current_timestamp,
    epsilon_greedy_policy,
)

from replay_buffers.utils import update_per_beta
from modules.utils import get_optimal_device


sys.path.append("../../")

from agents.rainbow_dqn import RainbowAgent


import random
from agents.agent import MARLBaseAgent
from agent_configs.dqn.nfsp_config import NFSPDQNConfig
from agents.policy_imitation import PolicyImitationAgent


class NFSPDQN(MARLBaseAgent):
    def __init__(
        self,
        env,
        config: NFSPDQNConfig,
        name: str = f"nfsp_dqn_{current_timestamp():.1f}",
        device: torch.device = get_optimal_device(),
    ) -> None:
        super().__init__(env, config, name)
        rl_configs = self.config.rl_configs
        sl_configs = self.config.sl_configs

        if self.config.shared_networks_and_buffers:
            rl_agent = RainbowAgent(
                env,
                rl_configs[0],
                f"rl_agent_{name}_0",
                device,
                num_players=self.config.game.num_players,
            )
            self.rl_agents = [rl_agent] * self.config.game.num_players
            if self.config.anticipatory_param != 1.0:
                sl_agent = PolicyImitationAgent(
                    env, sl_configs[0], f"sl_agent_{name}_0", device
                )
                self.sl_agents = [sl_agent] * self.config.game.num_players
        else:
            self.rl_agents: list[RainbowAgent] = [
                RainbowAgent(env, rl_configs[p], f"rl_agent_{name}_{p}", device)
                for p in range(self.config.game.num_players)
            ]
            if self.config.anticipatory_param != 1.0:
                self.sl_agents: list[PolicyImitationAgent] = [
                    PolicyImitationAgent(
                        env, sl_configs[p], f"sl_agent_{name}_{p}", device
                    )
                    for p in range(self.config.game.num_players)
                ]
        self.policies = ["average_strategy"] * self.config.game.num_players

        self.previous_states = [None] * self.config.game.num_players
        self.previous_infos = [None] * self.config.game.num_players
        self.previous_actions = [None] * self.config.game.num_players

        test_score_keys = [
            "test_score_vs_{}".format(agent.model_name) for agent in self.test_agents
        ]

        self.stats = StatTracker(
            model_name=self.model_name,
            stat_keys=[
                "rl_loss",
                "sl_loss",
                "test_score",
            ]
            + test_score_keys,
            target_values={},
            use_tensor_dicts={
                **{
                    key: ["score"]
                    + [
                        "player_{}_score".format(player)
                        for player in range(self.config.game.num_players)
                    ]
                    + [
                        "player_{}_win%".format(player)
                        for player in range(self.config.game.num_players)
                    ]
                    for key in test_score_keys
                },
            },
        )
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types("rl_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("sl_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.start_training_step = 0

    def select_agent_policies(self):
        for p in range(self.config.game.num_players):
            # print("Selecting policy")
            if random.random() <= self.config.anticipatory_param:
                # print("best_response")
                self.policies[p] = "best_response"
            else:
                # print("average_strategy")
                self.policies[p] = "average_strategy"

    def predict(self, state, info, *args, **kwargs):
        if self.policies[info["player"]] == "average_strategy":
            prediction = self.sl_agents[info["player"]].predict(state, info)
        else:
            prediction = self.rl_agents[info["player"]].predict(state)
        return prediction

    def select_actions(self, prediction, info: dict) -> torch.Tensor:
        if self.policies[info["player"]] == "average_strategy":
            # print("selecting with average strategy")
            action = self.sl_agents[info["player"]].select_actions(prediction)
        else:
            # print("selecting with best response")
            action = self.rl_agents[info["player"]].select_actions(
                prediction, info=info
            )
        return action

    def learn(self):
        rl_losses = []
        sl_losses = []
        for p in (
            [0]
            if self.config.shared_networks_and_buffers
            else range(self.config.game.num_players)
        ):
            if (
                self.rl_agents[p].replay_buffer.size
                >= self.rl_agents[p].config.min_replay_buffer_size
            ):  # only do if not in rl agent mode? (open spiel)
                rl_loss = self.rl_agents[p].learn().mean()
                rl_losses.append(rl_loss)
            if (
                self.config.anticipatory_param != 1.0
                and self.sl_agents[p].replay_buffer.size
                >= self.sl_agents[p].config.min_replay_buffer_size
            ):
                sl_loss = self.sl_agents[p].learn()
                sl_losses.append(sl_loss)

        average_rl_loss = (
            sum(rl_losses) / len(rl_losses) if len(rl_losses) > 0 else None
        )
        average_sl_loss = (
            sum(sl_losses) / len(sl_losses) if len(sl_losses) > 0 else None
        )
        return average_rl_loss, average_sl_loss

    def store_transition(self, action, state, info, reward, done, player):
        previous_state = self.previous_states[player]
        previous_info = self.previous_infos[player]
        previous_action = self.previous_actions[player]

        if (
            previous_action is not None
            and previous_state is not None
            and previous_info is not None
        ):
            assert (
                self.previous_infos[player]["player"] == info["player"] or done == True
            ), "Players don't match, {} != {}".format(
                self.previous_infos[player]["player"], info["player"]
            )
            transition = [
                previous_state,
                previous_info,
                previous_action,
                reward,
                state,
                info,
                done,
            ]

            # only do this if it is average policy? (open spiel)
            self.rl_agents[player].replay_buffer.store(
                *transition,
                player=player if self.config.shared_networks_and_buffers else 0,
            )

        if not done:
            # print("experience cached for next step")
            self.previous_states[player] = copy.deepcopy(state)
            self.previous_infos[player] = copy.deepcopy(info)
            self.previous_actions[player] = copy.deepcopy(action)
        else:
            # print("reset experience cache")
            self.previous_states[player] = None
            self.previous_infos[player] = None
            self.previous_actions[player] = None

    def fill_replay_buffers(self):
        pass

    def train(self):
        self.select_agent_policies()
        print(f"🎯 Initial policies: {self.policies}")

        # Initialize environment properly for PettingZoo
        self.env.reset()
        # Get initial state for first agent
        state, reward, termination, truncation, info = self.env.last()
        done = termination or truncation
        agent_id = self.env.agent_selection
        current_player = self.env.agents.index(agent_id)
        for _ in range(self.start_training_step, self.config.training_steps):
            # print(training_step)
            with torch.no_grad():
                for replay_step in range(self.config.replay_interval):
                    # print("Replay step:", replay_step)
                    # Handle agent turn in PettingZoo
                    # Get current agent and player index

                    prediction = self.predict(state, info)
                    if self.policies[current_player] == "best_response":
                        action = epsilon_greedy_policy(
                            prediction,
                            info,
                            self.rl_agents[current_player].eg_epsilon,
                            wrapper=lambda prediction, info: self.select_actions(
                                prediction, info
                            ).item(),
                        )
                    else:
                        action = self.select_actions(
                            prediction,
                            info,
                        ).item()

                    # print(f"player {current_player} action: {action}")
                    # Store transition for RL agent using current step reward
                    self.store_transition(
                        action,
                        state,
                        info,
                        reward,
                        done,
                        current_player,
                    )

                    # Store behavior for supervised learning if using best response
                    if (
                        self.policies[current_player] == "best_response"
                        and self.config.anticipatory_param != 1.0
                    ):
                        target_policy = torch.zeros(self.num_actions)
                        target_policy[action] = 1.0
                        self.sl_agents[current_player].replay_buffer.store(
                            state, info, target_policy
                        )

                    # Step environment
                    self.env.step(action)

                    (
                        next_state,
                        reward,
                        termination,
                        truncation,
                        next_info,
                    ) = self.env.last()
                    done = termination or truncation
                    state = next_state
                    info = next_info
                    agent_id = self.env.agent_selection
                    current_player = self.env.agents.index(agent_id)
                    if done:
                        # Store final transitions for all players before reset
                        for p in range(self.config.game.num_players):
                            self.store_transition(
                                action,  # not stored
                                state,  # not stored
                                info,  # not stored
                                self.env.rewards[self.env.agents[p]],  # stored
                                done,  # stored
                                p,
                            )

                        # Reset environment and get new episode
                        self.env.reset()
                        state, reward, termination, truncation, info = self.env.last()
                        done = termination or truncation
                        agent_id = self.env.agent_selection
                        current_player = self.env.agents.index(agent_id)
                        self.select_agent_policies()

                        # Reset previous states for new episode
                        self.previous_states = [None] * self.config.game.num_players
                        self.previous_infos = [None] * self.config.game.num_players
                        self.previous_actions = [None] * self.config.game.num_players

            for p in (
                [0]
                if self.config.shared_networks_and_buffers
                else range(self.config.game.num_players)
            ):
                old_epsilon = self.rl_agents[p].eg_epsilon
                self.rl_agents[p].update_eg_epsilon(self.training_step)
                if self.training_step % 1000 == 0 and p == 0:
                    print(
                        f"   Player {p} ε: {old_epsilon:.4f} → {self.rl_agents[p].eg_epsilon:.4f}"
                    )

                self.rl_agents[p].replay_buffer.set_beta(
                    update_per_beta(
                        self.rl_agents[p].replay_buffer.beta,
                        self.rl_agents[p].config.per_beta_final,
                        self.config.training_steps,
                        self.rl_agents[p].config.per_beta,
                    )
                )
            for minibatch in range(self.config.num_minibatches):
                rl_loss, sl_loss = self.learn()
                if rl_loss is not None:
                    self.stats.append("rl_loss", rl_loss)
                    print(f"   RL loss: {rl_loss:.6f}")
                if sl_loss is not None:
                    self.stats.append("sl_loss", sl_loss)
                    print(f"   SL loss: {sl_loss:.6f}")
                if rl_loss is not None or sl_loss is not None:
                    self.training_step += 1

            # Update target networks
            if self.training_step % self.rl_agents[0].config.transfer_interval == 0:
                # print(f"🎯 Updating target networks at step {training_step}")
                for p in (
                    [0]
                    if self.config.shared_networks_and_buffers
                    else range(self.config.game.num_players)
                ):
                    self.rl_agents[p].update_target_model()

            # Buffer size monitoring
            if self.training_step % 1000 == 0:
                print(f"Buffer sizes at step {self.training_step}:")
                for p in range(min(2, self.config.game.num_players)):
                    rl_size = self.rl_agents[p].replay_buffer.size
                    rl_capacity = self.rl_agents[p].replay_buffer.max_size
                    print(f"   Player {p} RL buffer: {rl_size}/{rl_capacity}")

                    if self.config.anticipatory_param != 1.0:
                        sl_size = self.sl_agents[p].replay_buffer.size
                        sl_capacity = self.sl_agents[p].replay_buffer.max_size
                        print(f"   Player {p} SL buffer: {sl_size}/{sl_capacity}")

            # Checkpointing
            if (
                self.training_step % self.checkpoint_interval == 0
                and self.training_step > self.start_training_step
            ):

                print("P1 SL Buffer Size: ", self.sl_agents[0].replay_buffer.size)
                # p1_actions_distribution = [0] * self.num_actions
                # for i in range(0, self.sl_agents[0].replay_buffer.size):
                #     p1_actions_distribution += self.sl_agents[
                #         0
                #     ].replay_buffer.target_policy_buffer[i]

                # print("P1 SL buffer distribution", p1_actions_distribution)
                # p1_actions_distribution = np.array(p1_actions_distribution)
                # p1_actions_distribution = p1_actions_distribution / np.sum(
                #     p1_actions_distribution
                # )
                # print("P1 actions distribution", p1_actions_distribution)

                print("P2 SL Buffer Size: ", self.sl_agents[1].replay_buffer.size)
                # p2_actions_distribution = [0] * self.num_actions
                # for i in range(0, self.sl_agents[1].replay_buffer.size):
                #     p2_actions_distribution += self.sl_agents[
                #         1
                #     ].replay_buffer.target_policy_buffer[i]

                # print("P2 SL buffer distribution", p2_actions_distribution)
                # p2_actions_distribution = np.array(p2_actions_distribution)
                # p2_actions_distribution = p2_actions_distribution / np.sum(
                #     p2_actions_distribution
                # )
                # print("P2 actions distribution", p2_actions_distribution)
                self.run_tests(self.stats)
                self.save_checkpoint(save_weights=self.config.save_intermediate_weights)

        self.save_checkpoint(save_weights=True)
        self.env.close()

    def load_optimizer_state(self, checkpoint):
        for p in (
            [0]
            if self.config.shared_networks_and_buffers
            else range(self.config.game.num_players)
        ):
            self.rl_agents[p].optimizer.load_state_dict(
                checkpoint["optimizer_{}".format(p)]
            )
            if self.config.anticipatory_param != 1.0:
                self.sl_agents[p].optimizer.load_state_dict(
                    checkpoint["sl_optimizer_{}".format(p)]
                )

    def load_replay_buffers(self, checkpoint):
        for p in (
            [0]
            if self.config.shared_networks_and_buffers
            else range(self.config.game.num_players)
        ):
            self.rl_agents[p].replay_buffer = checkpoint["replay_buffer_{}".format(p)]
            if self.config.anticipatory_param != 1.0:
                self.sl_agents[p].replay_buffer = checkpoint[
                    "sl_replay_buffer_{}".format(p)
                ]

    def load_model_weights(self, checkpoint):
        for p in (
            [0]
            if self.config.shared_networks_and_buffers
            else range(self.config.game.num_players)
        ):
            self.rl_agents[p].model.load_state_dict(checkpoint["model_{}".format(p)])
            if self.config.anticipatory_param != 1.0:
                self.sl_agents[p].model.load_state_dict(
                    checkpoint["sl_model_{}".format(p)]
                )

    def checkpoint_environment(self, checkpoint):
        print(
            "WARNING: NFSP does not checkpoint environments, as RL card environments are not pickleable"
        )
        return checkpoint

    def checkpoint_optimizer_state(self, checkpoint):
        for p in (
            [0]
            if self.config.shared_networks_and_buffers
            else range(self.config.game.num_players)
        ):
            checkpoint["optimizer_{}".format(p)] = self.rl_agents[
                p
            ].optimizer.state_dict()
            if self.config.anticipatory_param != 1.0:
                checkpoint["sl_optimizer_{}".format(p)] = self.sl_agents[
                    p
                ].optimizer.state_dict()
        return checkpoint

    def checkpoint_replay_buffers(self, checkpoint):
        for p in (
            [0]
            if self.config.shared_networks_and_buffers
            else range(self.config.game.num_players)
        ):
            checkpoint["replay_buffer_{}".format(p)] = self.rl_agents[p].replay_buffer
            if self.config.anticipatory_param != 1.0:
                checkpoint["sl_replay_buffer_{}".format(p)] = self.sl_agents[
                    p
                ].replay_buffer
        return checkpoint

    def checkpoint_model_weights(self, checkpoint):
        for p in (
            [0]
            if self.config.shared_networks_and_buffers
            else range(self.config.game.num_players)
        ):
            checkpoint["model_{}".format(p)] = self.rl_agents[p].model.state_dict()
            if self.config.anticipatory_param != 1.0:
                checkpoint["sl_model_{}".format(p)] = self.sl_agents[
                    p
                ].model.state_dict()
        return checkpoint

    def test(self, num_trials, player=None, dir="./checkpoints"):
        """
        Test the trained NFSP agent against itself or specific players

        Args:
            num_trials: Number of episodes to test
            player: Specific player to test (None for all players)
            step: Training step for logging
            dir: Directory for saving results
        """
        final_exploitability = 0
        for player in range(self.config.game.num_players):
            # Set test agent to average strategy, and all other agents to best response (see how much a best response can exploit the average strategy)
            original_policies = self.policies.copy()
            self.policies = ["best_response"] * self.config.game.num_players
            self.policies[player] = (
                "average_strategy"
                if self.config.anticipatory_param != 1.0
                else "best_response"
            )

            test_results = super().test(num_trials // 2, player, dir)
            exploitability = -test_results["score"]

            # Restore original policies
            self.policies = original_policies
            final_exploitability += exploitability
        final_exploitability /= self.config.game.num_players
        return final_exploitability

    def load_from_checkpoint(agent_class, config_class, dir: str, training_step):
        # load the config and checkpoint
        training_step_dir = Path(dir, f"step_{training_step}")
        weights_dir = Path(training_step_dir, "model_weights")
        weights_path = str(Path(training_step_dir, f"model_weights/weights.keras"))
        config = config_class.load(Path(dir, "configs/config.yaml"))
        checkpoint = torch.load(weights_path)
        env = checkpoint["enviroment"]
        model_name = checkpoint["model_name"]

        # construct the agent
        agent = agent_class(env, config, model_name, from_checkpoint=True)

        # load the model state (weights, optimizer, replay buffer, training time, training step, total environment steps)
        os.makedirs(weights_dir, exist_ok=True)

        agent.training_time = checkpoint["training_time"]
        agent.training_step = checkpoint["training_step"]
        agent.total_environment_steps = checkpoint["total_environment_steps"]

        agent.load_model_weights(checkpoint)
        agent.load_optimizer_state(checkpoint)
        agent.load_replay_buffers(checkpoint)

        # load the graph stats and targets
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "rb") as f:
            agent.stats = pickle.load(f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "rb") as f:
            agent.targets = pickle.load(f)

        return agent
