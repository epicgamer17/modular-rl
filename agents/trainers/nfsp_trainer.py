"""
NFSPTrainer orchestrates the training process for NFSP by coordinating
the executor for data collection, the learner for optimization, and
handling the training loop.

Supports:
- Shared networks/buffers: All players share the same BR/AVG networks (default)
- Separate networks/buffers: Each player has their own BR/AVG networks and learners
"""

import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import dill as pickle

from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from agents.learners.nfsp_learner import NFSPLearner
from agents.policies.nfsp_policy import NFSPPolicy
from agents.action_selectors.selectors import EpsilonGreedy, CategoricalSelector
from agents.actors.actors import get_actor_class
from modules.agent_nets.rainbow_dqn import RainbowNetwork
from modules.agent_nets.policy_imitation import SupervisedNetwork
from stats.stats import StatTracker, PlotType


class NFSPTrainer:
    """
    NFSPTrainer orchestrates the NFSP training process.

    Supports both shared networks/buffers (all players share networks) and
    separate networks/buffers (each player has their own networks/learners).
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        """
        Initializes the NFSPTrainer.

        Args:
            config: NFSPDQNConfig with hyperparameters.
            env: Environment instance (not a factory function).
            device: Torch device for training.
            stats: Optional StatTracker for logging metrics.
            test_agents: Optional list of agents to test against.
        """
        self.config = config
        self.device = device
        self.stats = (
            stats if stats is not None else StatTracker(model_name=config.model_name)
        )
        self.test_agents = test_agents if test_agents is not None else []

        self._env = env
        assert not callable(
            env
        ), "env must be an environment instance, not a factory function"

        # Detect player IDs from environment
        if hasattr(env, "possible_agents") and len(env.possible_agents) > 0:
            self.player_ids = list(env.possible_agents)
        elif hasattr(env, "agents") and len(env.agents) > 0:
            self.player_ids = list(env.agents)
        else:
            self.player_ids = ["player_0"]

        self._player_id = self.player_ids[0]

        # Get observation/action specs
        obs_dim, obs_dtype = self._determine_observation_dimensions(env)
        # Use config override if provided, otherwise use auto-detected dtype
        if config.observation_dtype is not None:
            obs_dtype = config.observation_dtype
        num_actions = self._get_num_actions(env)
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.obs_dtype = obs_dtype

        # Use config field directly (no getattr)
        self.shared_networks = config.shared_networks_and_buffers

        if self.shared_networks:
            self._init_shared_networks(obs_dim, num_actions)
        else:
            self._init_separate_networks(obs_dim, num_actions)

        # Launch workers using config fields directly
        worker_args = (
            config.game.make_env,
            self.policy,
            config.game.num_players,
        )
        actor_cls = get_actor_class(env)
        self.executor.launch(actor_cls, worker_args, config.num_workers)

        self.training_step = 0
        self.model_name = config.model_name

        # Intervals
        total_steps = config.training_steps
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

    def _init_shared_networks(self, obs_dim, num_actions):
        """Initialize networks and learners for shared networks mode."""
        rl_config = self.config.rl_configs[0]
        sl_config = self.config.sl_configs[0]

        # RL Network (Best Response)
        self.br_model = RainbowNetwork(
            config=rl_config,
            input_shape=obs_dim,  # Exclude batch dim
            output_size=num_actions,
        ).to(self.device)

        self.br_target_model = RainbowNetwork(
            config=rl_config,
            input_shape=obs_dim,  # Exclude batch dim
            output_size=num_actions,
        ).to(self.device)
        self.br_target_model.load_state_dict(self.br_model.state_dict())

        # SL Network (Average Strategy)
        self.avg_model = SupervisedNetwork(
            config=sl_config,
            output_size=num_actions,
            input_shape=obs_dim,  # Exclude batch dim
        ).to(self.device)

        if self.config.multi_process:
            self.br_model.share_memory()
            self.br_target_model.share_memory()
            self.avg_model.share_memory()

        # Action Selectors
        self.br_selector = EpsilonGreedy(epsilon=rl_config.eg_epsilon)
        self.avg_selector = CategoricalSelector(from_logits=False)

        # Policy (shared networks mode)
        self.policy = NFSPPolicy(
            best_response_model=self.br_model,
            average_model=self.avg_model,
            best_response_selector=self.br_selector,
            average_selector=self.avg_selector,
            device=self.device,
            eta=self.config.anticipatory_param,
            player_ids=self.player_ids,
        )

        # Single learner for all players
        self.learner = NFSPLearner(
            config=self.config,
            best_response_model=self.br_model,
            best_response_target_model=self.br_target_model,
            average_model=self.avg_model,
            device=self.device,
            num_actions=num_actions,
            observation_dimensions=obs_dim,
            observation_dtype=self.obs_dtype,
        )
        self.learners = None  # Not used in shared mode

        # Executor based on config
        if self.config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

    def _init_separate_networks(self, obs_dim, num_actions):
        """Initialize networks and learners for separate networks per player."""
        rl_config = self.config.rl_configs[0]
        sl_config = self.config.sl_configs[0]

        # Per-player networks
        self.br_models: Dict[str, RainbowNetwork] = {}
        self.br_target_models: Dict[str, RainbowNetwork] = {}
        self.avg_models: Dict[str, SupervisedNetwork] = {}
        self.learners: Dict[str, NFSPLearner] = {}

        for player_id in self.player_ids:
            # BR Network
            br_model = RainbowNetwork(
                config=rl_config,
                input_shape=obs_dim,  # Exclude batch dim
                output_size=num_actions,
            ).to(self.device)
            br_target = RainbowNetwork(
                config=rl_config,
                input_shape=obs_dim,  # Exclude batch dim
                output_size=num_actions,
            ).to(self.device)
            br_target.load_state_dict(br_model.state_dict())

            # AVG Network
            avg_model = SupervisedNetwork(
                config=sl_config,
                output_size=num_actions,
                input_shape=obs_dim,  # Exclude batch dim
            ).to(self.device)

            if self.config.multi_process:
                br_model.share_memory()
                br_target.share_memory()
                avg_model.share_memory()

            self.br_models[player_id] = br_model
            self.br_target_models[player_id] = br_target
            self.avg_models[player_id] = avg_model

            # Per-player learner
            self.learners[player_id] = NFSPLearner(
                config=self.config,
                best_response_model=br_model,
                best_response_target_model=br_target,
                average_model=avg_model,
                device=self.device,
                num_actions=num_actions,
                observation_dimensions=obs_dim,
                observation_dtype=self.obs_dtype,
            )

        # Action Selectors (shared)
        self.br_selector = EpsilonGreedy(epsilon=rl_config.eg_epsilon)
        self.avg_selector = CategoricalSelector(from_logits=False)

        # Policy (separate networks mode)
        self.policy = NFSPPolicy(
            best_response_models=self.br_models,
            average_models=self.avg_models,
            best_response_selector=self.br_selector,
            average_selector=self.avg_selector,
            device=self.device,
            eta=self.config.anticipatory_param,
        )

        # Executor based on config
        if self.config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

    def train(self) -> None:
        """Main training loop."""
        self._setup_stats()

        print(f"Starting NFSP training for {self.config.training_steps} steps...")
        print(
            f"Mode: {'Shared networks' if self.shared_networks else 'Separate networks per player'}"
        )

        last_log_time = time.time()

        while self.training_step < self.config.training_steps:
            # 1. Update worker weights and parameters
            self._update_worker_weights()

            # 2. Collect data from workers
            sequences, collection_stats = self.executor.collect_data(
                self.config.replay_interval
            )

            # Log collection stats
            for key, val in collection_stats.items():
                self.stats.append(key, val)

            # 3. Store transitions
            for sequence in sequences:
                self._store_sequence_transitions(sequence)

            # 4. Learning step
            for _ in range(self.config.num_minibatches):
                self._learning_step()

            self.training_step += 1

            # Periodic logging
            if self.training_step % 10 == 0:
                elapsed = time.time() - last_log_time
                avg_score = 0.0
                if "score" in self.stats.stats and self.stats.stats["score"]:
                    avg_score = np.mean(self.stats.stats["score"][-10:])

                print(
                    f"Step {self.training_step}/{self.config.training_steps}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Time/10 steps: {elapsed:.2f}s"
                )
                last_log_time = time.time()
                self.stats.drain_queue()

            # 5. Update Target Networks (periodically)
            if self.training_step % self.config.rl_configs[0].transfer_interval == 0:
                self._update_target_networks()

            # 6. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 7. Periodic testing
            if self.training_step % self.test_interval == 0:
                self._run_tests()

        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def _update_worker_weights(self):
        """Update worker weights based on mode."""
        if self.shared_networks:
            self.executor.update_weights(
                {
                    "best_response_state_dict": self.br_model.state_dict(),
                    "average_state_dict": self.avg_model.state_dict(),
                    "eta": self.config.anticipatory_param,
                }
            )
        else:
            self.executor.update_weights(
                {
                    "best_response_state_dicts": {
                        pid: model.state_dict() for pid, model in self.br_models.items()
                    },
                    "average_state_dicts": {
                        pid: model.state_dict()
                        for pid, model in self.avg_models.items()
                    },
                    "eta": self.config.anticipatory_param,
                }
            )

    def _store_sequence_transitions(self, sequence):
        """
        Store sequence transitions in appropriate learner(s).

        For multi-player sequences, this method folds transitions so each player
        learns from their own observation sequence and correctly accumulated rewards.
        """
        if not sequence.policy_history:
            return

        # 1. Compute per-player accumulated rewards from all_player_rewards history
        if self.config.game.num_players > 1:
            player_rewards = self._compute_player_rewards(sequence)
        else:
            player_rewards = None

        # 2. Store transitions, folding the next_state to the player's next turn
        for i in range(len(sequence.action_history)):
            obs = sequence.observation_history[i]
            info = sequence.info_history[i]
            action = sequence.action_history[i]

            if sequence.player_id_history and i < len(sequence.player_id_history):
                player_id = sequence.player_id_history[i]
            else:
                player_id = self.player_ids[0]

            # Find NEXT turn for this specific player to get the correct next_observation
            next_turn_idx = -1
            if self.config.game.num_players > 1:
                for j in range(i + 1, len(sequence.action_history)):
                    if sequence.player_id_history[j] == player_id:
                        next_turn_idx = j
                        break

            if next_turn_idx != -1:
                # Next state for this player is the state at their next turn
                next_obs = sequence.observation_history[next_turn_idx]
                next_info = sequence.info_history[next_turn_idx]
                done = False
            else:
                # No more turns for this player in this episode - use terminal state
                next_obs = sequence.observation_history[-1]
                next_info = sequence.info_history[-1]
                done = True

            # Get correctly attributed reward
            if player_rewards is not None and player_id in player_rewards:
                reward = player_rewards[player_id].get(i, 0.0)
            else:
                reward = sequence.rewards[i]

            if i < len(sequence.policy_history):
                policy_used = sequence.policy_history[i]
            else:
                policy_used = "average_strategy"

            if self.shared_networks:
                self.learner.store(
                    observation=obs,
                    info=info,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    next_info=next_info,
                    done=done,
                    policy_used=policy_used,
                )
            else:
                # Route to player-specific learner
                if player_id in self.learners:
                    self.learners[player_id].store(
                        observation=obs,
                        info=info,
                        action=action,
                        reward=reward,
                        next_observation=next_obs,
                        next_info=next_info,
                        done=done,
                        policy_used=policy_used,
                    )

    def _compute_player_rewards(self, sequence) -> Dict[str, Dict[int, float]]:
        """
        Compute per-player accumulated rewards for turnover-based sequences.

        Uses 'all_player_rewards' from info history to ensure rewards occurring
        between a player's turns are correctly captured.
        """
        player_rewards: Dict[str, Dict[int, float]] = {
            pid: {} for pid in self.player_ids
        }

        # Track the index of the LAST action taken by each player
        last_action_idx: Dict[str, int] = {}

        for i in range(len(sequence.action_history)):
            if sequence.player_id_history and i < len(sequence.player_id_history):
                acting_player = sequence.player_id_history[i]
            else:
                acting_player = self.player_ids[0]

            # The rewards in info_history[i+1] were generated by action i (acting_player)
            # We attribute rewards to EACH player for their PREVIOUS move.
            summary_info = sequence.info_history[i + 1]
            all_rewards = summary_info.get("all_player_rewards", {})

            for pid, r in all_rewards.items():
                if pid in last_action_idx:
                    idx = last_action_idx[pid]
                    player_rewards[pid][idx] = player_rewards[pid].get(idx, 0.0) + r
                else:
                    # Initial reward? (e.g. at start of sequence)
                    # We can't attribute it to an action if player hasn't moved yet.
                    pass

            # Record this as the latest action for the acting player
            last_action_idx[acting_player] = i

        return player_rewards

    def _learning_step(self):
        """Perform learning step(s)."""
        if self.shared_networks:
            loss_stats = self.learner.step(self.stats)
            if loss_stats:
                for key, val in loss_stats.items():
                    self.stats.append(key, val)
        else:
            # Train each player's learner
            for player_id, learner in self.learners.items():
                loss_stats = learner.step(self.stats)
                if loss_stats:
                    for key, val in loss_stats.items():
                        self.stats.append(f"{player_id}_{key}", val)

    def _update_target_networks(self):
        """Update target networks."""
        if self.shared_networks:
            self.learner.update_target_network()
        else:
            for learner in self.learners.values():
                learner.update_target_network()

    def _run_tests(self) -> None:
        """Runs evaluation episodes and exploitability tests."""
        dir_path = Path("checkpoints", self.model_name)
        step_dir = Path(dir_path, f"step_{self.training_step}")

        # Standard AVG vs AVG test
        test_results = self.test(self.test_trials, dir=step_dir)
        if test_results:
            self.stats.append("test_score", test_results.get("score"))
            print(f"Test score: {test_results.get('score'):.3f}")

        # Exploitability tests (AVG vs BR matchups)
        exploit_results = self.test_exploitability(num_trials=self.test_trials)
        if exploit_results:
            # Log per-player BR payoffs as subkeys of 'exploitability' for combined plot
            for pid in self.player_ids:
                br_key = f"{pid}_br_vs_avg"
                if br_key in exploit_results:
                    # Use subkey format: append to 'exploitability' with subkey
                    self.stats.append(
                        "exploitability", exploit_results[br_key], subkey=f"{pid}_br"
                    )

            # Log total exploitability
            total = exploit_results.get("exploitability", 0.0)
            self.stats.append("exploitability", total, subkey="total")

            # Print summary
            print(f"Exploitability (sum of BR payoffs): {total:.3f}")

    def test(self, num_trials: int, dir: str = "./checkpoints") -> Dict[str, float]:
        """Runs evaluation episodes."""
        if num_trials == 0:
            return {}

        test_env = self.config.game.make_env()
        scores = []

        # For testing, use the Average Strategy (fully mixed)
        original_eta = self.policy.eta
        self.policy.eta = 0.0  # Force average strategy

        num_players = self.config.game.num_players

        with torch.inference_mode():
            for _ in range(num_trials):
                # 1. Reset
                if num_players != 1:
                    test_env.reset()
                    state, reward, terminated, truncated, info = test_env.last()
                else:
                    state, info = test_env.reset()

                self.policy.reset(state)
                done = False
                episode_reward = 0.0

                while not done:
                    # Get player_id for multi-player
                    if num_players != 1 and hasattr(test_env, "agent_selection"):
                        player_id = test_env.agent_selection
                    else:
                        player_id = None

                    action = self.policy.compute_action(
                        state, info, player_id=player_id
                    )
                    action_val = action.item() if torch.is_tensor(action) else action

                    # 2. Step
                    if num_players != 1:
                        test_env.step(action_val)
                        state, reward, terminated, truncated, info = test_env.last()
                        episode_reward += float(test_env.rewards[self._player_id])
                    else:
                        state, reward, terminated, truncated, info = test_env.step(
                            action_val
                        )
                        episode_reward += float(reward)

                    done = terminated or truncated

                scores.append(episode_reward)

        test_env.close()
        self.policy.eta = original_eta  # Restore

        return {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }

    def test_exploitability(self, num_trials: int = 10) -> Dict[str, float]:
        """
        Tests AVG vs BR matchups for each player to measure exploitability.

        For 2-player games, this runs:
        1. Player 1 AVG vs Player 2 BR -> measures how much P2 can exploit P1's avg strategy
        2. Player 1 BR vs Player 2 AVG -> measures how much P1 can exploit P2's avg strategy

        The sum of these payoffs is a measure of Nash convergence (lower = closer to Nash).
        At Nash equilibrium, best response should not gain advantage over average strategy.

        Args:
            num_trials: Number of episodes per matchup.

        Returns:
            Dict with per-player BR payoffs and exploitability sum.
        """
        if num_trials == 0 or len(self.player_ids) < 2:
            return {}

        results = {}

        # For each player, test their BR against opponent's AVG
        for br_player_id in self.player_ids:
            avg_player_ids = [pid for pid in self.player_ids if pid != br_player_id]

            br_payoff = self._run_cross_play_test(
                br_player_id=br_player_id,
                avg_player_ids=avg_player_ids,
                num_trials=num_trials,
            )

            results[f"{br_player_id}_br_vs_avg"] = br_payoff

        # Compute exploitability (sum of BR payoffs)
        # In a zero-sum game at Nash, each BR should get ~0 extra value
        exploitability = sum(results.values())
        results["exploitability"] = exploitability

        return results

    def _run_cross_play_test(
        self,
        br_player_id: str,
        avg_player_ids: List[str],
        num_trials: int,
    ) -> float:
        """
        Runs episodes where one player uses BR and others use AVG.

        Args:
            br_player_id: The player using best response.
            avg_player_ids: Players using average strategy.
            num_trials: Number of episodes.

        Returns:
            Average payoff for the BR player.
        """
        test_env = self.config.game.make_env()
        br_payoffs = []

        with torch.inference_mode():
            for _ in range(num_trials):
                # Reset
                test_env.reset()
                state, reward, terminated, truncated, info = test_env.last()

                # Force policy modes for this episode
                self.policy.current_policy = {}
                self.policy.current_policy[br_player_id] = "best_response"
                for avg_pid in avg_player_ids:
                    self.policy.current_policy[avg_pid] = "average_strategy"

                done = False
                episode_br_reward = 0.0

                while not done:
                    player_id = test_env.agent_selection

                    action = self.policy.compute_action(
                        state, info, player_id=player_id
                    )
                    action_val = action.item() if torch.is_tensor(action) else action

                    test_env.step(action_val)
                    state, reward, terminated, truncated, info = test_env.last()

                    # Track BR player's reward
                    episode_br_reward += float(test_env.rewards.get(br_player_id, 0.0))

                    done = terminated or truncated

                br_payoffs.append(episode_br_reward)

        test_env.close()

        return sum(br_payoffs) / len(br_payoffs) if br_payoffs else 0.0

    def _save_checkpoint(self) -> None:
        """Saves model weights and stats."""
        base_dir = Path("checkpoints", self.model_name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        weights_dir = step_dir / "model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = weights_dir / "weights.pt"

        if self.shared_networks:
            checkpoint = {
                "training_step": self.training_step,
                "model_name": self.model_name,
                "shared_networks": True,
                "br_model": self.br_model.state_dict(),
                "avg_model": self.avg_model.state_dict(),
                "rl_optimizer": self.learner.rl_learner.optimizer.state_dict(),
                "sl_optimizer": self.learner.sl_optimizer.state_dict(),
            }
        else:
            checkpoint = {
                "training_step": self.training_step,
                "model_name": self.model_name,
                "shared_networks": False,
                "br_models": {pid: m.state_dict() for pid, m in self.br_models.items()},
                "avg_models": {
                    pid: m.state_dict() for pid, m in self.avg_models.items()
                },
                "rl_optimizers": {
                    pid: l.rl_learner.optimizer.state_dict()
                    for pid, l in self.learners.items()
                },
                "sl_optimizers": {
                    pid: l.sl_optimizer.state_dict() for pid, l in self.learners.items()
                },
            }
        torch.save(checkpoint, weights_path)

        if hasattr(self, "stats"):
            stats_dir = step_dir / "graphs_stats"
            os.makedirs(stats_dir, exist_ok=True)
            with open(stats_dir / "stats.pkl", "wb") as f:
                pickle.dump(self.stats.get_data(), f)
            graph_dir = base_dir / "graphs"
            os.makedirs(graph_dir, exist_ok=True)
            self.stats.plot_graphs(dir=graph_dir)

    def _determine_observation_dimensions(self, env):
        """Infers input dimensions for the neural network."""
        import gymnasium as gym

        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Box):
            return torch.Size(obs_space.shape), obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return torch.Size((1,)), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return torch.Size((len(obs_space.spaces),)), np.int32
        elif callable(obs_space):
            # For PettingZoo-style callable observation spaces
            try:
                space = obs_space(self._player_id)
            except KeyError:
                if hasattr(env, "possible_agents") and env.possible_agents:
                    space = obs_space(env.possible_agents[0])
                else:
                    raise
            return torch.Size(space.shape), space.dtype
        else:
            return torch.Size(obs_space.shape), obs_space.dtype

    def _get_num_actions(self, env) -> int:
        """Determines action space properties."""
        import gymnasium as gym

        if isinstance(env.action_space, gym.spaces.Discrete):
            return int(env.action_space.n)
        elif callable(env.action_space):  # PettingZoo
            return int(env.action_space(self._player_id).n)
        elif hasattr(self.config.game, "num_actions"):
            return self.config.game.num_actions
        else:
            # Box/Continuous
            return int(env.action_space.shape[0])

    def _setup_stats(self) -> None:
        # Basic stat keys
        basic_keys = ["score", "rl_loss", "sl_loss", "sl_policy", "test_score"]
        for key in basic_keys:
            if key not in self.stats.stats:
                self.stats._init_key(key)

        # Exploitability uses subkeys so all lines appear on the same plot
        # Subkeys: {pid}_br for each player's BR payoff, 'total' for sum
        exploit_subkeys = [f"{pid}_br" for pid in self.player_ids] + ["total"]
        if "exploitability" not in self.stats.stats:
            self.stats._init_key("exploitability", subkeys=exploit_subkeys)

        self.stats.add_plot_types("score", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types("exploitability", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types("sl_policy", PlotType.BAR, max_bars=20)
