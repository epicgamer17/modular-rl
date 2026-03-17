import torch
import time
from typing import Optional, List, Dict, Any, Tuple

from agents.trainers.base_trainer import BaseTrainer
from agents.learners.nfsp_learner import NFSPLearner
from agents.action_selectors.selectors import (
    EpsilonGreedySelector,
    CategoricalSelector,
    NFSPSelector,
)
from agents.workers.actors import get_actor_class
from agents.executors.local_executor import LocalExecutor
from agents.executors.torch_mp_executor import TorchMPExecutor
from replay_buffers.transition import TransitionBatch, Transition
from modules.agent_nets.modular import ModularAgentNetwork
from stats.stats import StatTracker, PlotType
import numpy as np


class NFSPTrainer(BaseTrainer):
    """
    NFSPTrainer orchestrates the NFSP training process.
    """

    def __init__(
        self,
        config,
        env,
        device: torch.device,
        name: str = "agent",
        stats: Optional[StatTracker] = None,
        test_agents: Optional[List] = None,
    ):
        super().__init__(config, env, device, name, stats, test_agents)

        self.shared_networks = config.shared_networks_and_buffers
        self.player_ids = list(env.possible_agents)

        if self.shared_networks:
            self._init_shared_networks(self.obs_dim, self.num_actions)
        else:
            self._init_separate_networks(self.obs_dim, self.num_actions)

    def _init_shared_networks(self, obs_dim, num_actions):
        """Initialize networks and learners for shared networks mode."""
        rl_config = self.config.rl_configs[0]
        sl_config = self.config.sl_configs[0]

        # RL Network (Best Response)
        self.br_agent_network = ModularAgentNetwork(
            config=rl_config,
            input_shape=obs_dim,  # Exclude batch dim
            num_actions=num_actions,
        ).to(self.device)

        self.br_target_agent_network = ModularAgentNetwork(
            config=rl_config,
            input_shape=obs_dim,  # Exclude batch dim
            num_actions=num_actions,
        ).to(self.device)
        # Sync target network (cleaning state dict in case agent_network is compiled)
        from modules.utils import get_clean_state_dict

        clean_state = get_clean_state_dict(self.br_agent_network)
        self.br_target_agent_network.load_state_dict(clean_state, strict=False)

        # SL Network (Average Strategy)
        self.avg_agent_network = ModularAgentNetwork(
            config=sl_config,
            num_actions=num_actions,
            input_shape=obs_dim,  # Exclude batch dim
        ).to(self.device)

        if self.config.multi_process:
            self.br_agent_network.share_memory()
            self.br_target_agent_network.share_memory()
            self.avg_agent_network.share_memory()

        # Action Selectors
        self.br_selector = EpsilonGreedySelector(epsilon=rl_config.eg_epsilon)
        self.avg_selector = CategoricalSelector(exploration=True)

        # Networks dict for the selector
        self.networks = torch.nn.ModuleDict(
            {
                "best_response": self.br_agent_network,
                "average_strategy": self.avg_agent_network,
            }
        )

        # NFSP Selector (shared networks mode)
        self.selector = NFSPSelector(
            br_selector=self.br_selector,
            avg_selector=self.avg_selector,
            eta=self.config.anticipatory_param,
        )

        # Single learner for all players
        self.learner = NFSPLearner(
            config=self.config,
            best_response_agent_network=self.br_agent_network,
            best_response_target_agent_network=self.br_target_agent_network,
            average_agent_network=self.avg_agent_network,
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

        worker_args = (
            self.config.game.make_env,
            self.networks,
            self.selector,
            None,  # Replay Buffer (handled by executor)
            self.num_players,
            self.config,
            self.device,
            self.name,
        )
        self.actor_cls = get_actor_class(self._env, self.config)

        self.executor.launch(self.actor_cls, worker_args, self.config.num_workers)

    def _init_separate_networks(self, obs_dim, num_actions):
        """Initialize networks and learners for separate networks per player."""
        rl_config = self.config.rl_configs[0]
        sl_config = self.config.sl_configs[0]

        # Per-player networks
        self.br_agent_networks: Dict[str, ModularAgentNetwork] = {}
        self.br_target_agent_networks: Dict[str, ModularAgentNetwork] = {}
        self.avg_agent_networks: Dict[str, ModularAgentNetwork] = {}
        self.learners: Dict[str, NFSPLearner] = {}

        for player_id in self.player_ids:
            # BR Network
            br_agent_network = ModularAgentNetwork(
                config=rl_config,
                input_shape=obs_dim,  # Exclude batch dim
                num_actions=num_actions,
            ).to(self.device)
            br_target_agent_network = ModularAgentNetwork(
                config=rl_config,
                input_shape=obs_dim,  # Exclude batch dim
                num_actions=num_actions,
            ).to(self.device)
            # Sync target network (cleaning state dict in case agent_network is compiled)
            from modules.utils import get_clean_state_dict

            clean_state = get_clean_state_dict(br_agent_network)
            br_target_agent_network.load_state_dict(clean_state, strict=False)

            # AVG Network
            avg_agent_network = ModularAgentNetwork(
                config=sl_config,
                num_actions=num_actions,
                input_shape=obs_dim,  # Exclude batch dim
            ).to(self.device)

            if self.config.multi_process:
                br_agent_network.share_memory()
                br_target_agent_network.share_memory()
                avg_agent_network.share_memory()

            self.br_agent_networks[player_id] = br_agent_network
            self.br_target_agent_networks[player_id] = br_target_agent_network
            self.avg_agent_networks[player_id] = avg_agent_network

            # Per-player learner
            self.learners[player_id] = NFSPLearner(
                config=self.config,
                best_response_agent_network=br_agent_network,
                best_response_target_agent_network=br_target_agent_network,
                average_agent_network=avg_agent_network,
                device=self.device,
                num_actions=num_actions,
                observation_dimensions=obs_dim,
                observation_dtype=self.obs_dtype,
            )

        # Action Selectors
        self.br_selector = EpsilonGreedySelector(epsilon=rl_config.eg_epsilon)
        self.avg_selector = CategoricalSelector(exploration=True)

        # Networks dict for selector
        self.networks = torch.nn.ModuleDict(
            {
                "best_response": torch.nn.ModuleDict(self.br_agent_networks),
                "average_strategy": torch.nn.ModuleDict(self.avg_agent_networks),
            }
        )

        # NFSP Selector (separate networks mode)
        self.selector = NFSPSelector(
            br_selector=self.br_selector,
            avg_selector=self.avg_selector,
            eta=self.config.anticipatory_param,
        )

        # Executor based on config
        if self.config.multi_process:
            self.executor = TorchMPExecutor()
        else:
            self.executor = LocalExecutor()

        worker_args = (
            self.config.game.make_env,
            self.networks,
            self.selector,
            None,  # Replay Buffer (handled by executor)
            self.num_players,
            self.config,
            self.device,
            self.name,
        )
        self.actor_cls = get_actor_class(self._env, self.config)

        self.executor.launch(self.actor_cls, worker_args, self.config.num_workers)

    def train(self) -> None:
        """Main training loop."""
        self.setup()

        print(f"Starting NFSP training for {self.config.training_steps} steps...")
        print(
            f"Mode: {'Shared networks' if self.shared_networks else 'Separate networks per player'}"
        )

        last_log_time = time.time()

        while self.training_step < self.config.training_steps:
            self.train_step()

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


            # 6. Periodic checkpointing
            if self.training_step % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 7. Periodic testing
            if self.training_step % self.test_interval == 0:
                self.trigger_test(state_dict={}, step=self.training_step)

        self.executor.stop()
        self._save_checkpoint()
        print("Training finished.")

    def train_step(self):
        """Single training step: collect data and perform optimization."""
        # 1. Update worker weights and parameters
        self._update_worker_weights()

        # 2. Collect data from workers
        sequences, collection_stats = self.executor.collect_data(
            min_samples=self.config.replay_interval,
            worker_type=self.actor_cls,
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

    def _update_worker_weights(self):
        """Update worker weights based on mode."""
        if self.shared_networks:
            self.executor.update_weights(
                {
                    "best_response_state_dict": self.br_agent_network.state_dict(),
                    "average_state_dict": self.avg_agent_network.state_dict(),
                    "eta": self.config.anticipatory_param,
                }
            )
        else:
            self.executor.update_weights(
                {
                    "best_response_state_dicts": {
                        pid: agent_network.state_dict()
                        for pid, agent_network in self.br_agent_networks.items()
                    },
                    "average_state_dicts": {
                        pid: agent_network.state_dict()
                        for pid, agent_network in self.avg_agent_networks.items()
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
            legal_moves = (
                sequence.legal_moves_history[i]
                if i < len(sequence.legal_moves_history)
                else []
            )
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
                next_legal_moves = (
                    sequence.legal_moves_history[next_turn_idx]
                    if next_turn_idx < len(sequence.legal_moves_history)
                    else []
                )
                done = False
            else:
                # No more turns for this player in this episode - use terminal state
                next_obs = sequence.observation_history[-1]
                next_legal_moves = (
                    sequence.legal_moves_history[-1]
                    if sequence.legal_moves_history
                    else []
                )
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
                    legal_moves=legal_moves,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    next_legal_moves=next_legal_moves,
                    done=done,
                    policy_used=policy_used,
                )
            else:
                # Route to player-specific learner
                if player_id in self.learners:
                    self.learners[player_id].store(
                        observation=obs,
                        legal_moves=legal_moves,
                        action=action,
                        reward=reward,
                        next_observation=next_obs,
                        next_legal_moves=next_legal_moves,
                        done=done,
                        policy_used=policy_used,
                    )

    def _compute_player_rewards(self, sequence) -> Dict[str, Dict[int, float]]:
        """
        Compute per-player accumulated rewards for turnover-based sequences.

        Uses 'all_player_rewards_history' to ensure rewards occurring
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

            # The rewards at index i+1 were generated by action i (acting_player)
            # We attribute rewards to EACH player for their PREVIOUS move.
            all_rewards = (
                sequence.all_player_rewards_history[i + 1]
                if i + 1 < len(sequence.all_player_rewards_history)
                else {}
            )

            for pid, r in all_rewards.items():
                if pid in last_action_idx:
                    idx = last_action_idx[pid]
                    player_rewards[pid][idx] = player_rewards[pid].get(idx, 0.0) + r
                else:
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
                    action_val = action.item()

                    test_env.step(action_val)
                    state, reward, terminated, truncated, info = test_env.last()

                    # Track BR player's reward
                    episode_br_reward += float(test_env.rewards.get(br_player_id, 0.0))

                    done = terminated or truncated

                br_payoffs.append(episode_br_reward)

        test_env.close()

        return sum(br_payoffs) / len(br_payoffs) if br_payoffs else 0.0

    def _save_checkpoint(self) -> None:
        """Saves NFSP checkpoint."""
        if self.shared_networks:
            checkpoint_data = {
                "shared_networks": True,
                "br_agent_network": self.br_agent_network.state_dict(),
                "avg_agent_network": self.avg_agent_network.state_dict(),
                "rl_optimizer": self.learner.rl_learner.optimizer.state_dict(),
                "sl_optimizer": self.learner.sl_optimizer.state_dict(),
            }
        else:
            checkpoint_data = {
                "shared_networks": False,
                "br_agent_networks": {
                    pid: an.state_dict() for pid, an in self.br_agent_networks.items()
                },
                "avg_agent_networks": {
                    pid: an.state_dict() for pid, an in self.avg_agent_networks.items()
                },
                "rl_optimizers": {
                    pid: l.rl_learner.optimizer.state_dict()
                    for pid, l in self.learners.items()
                },
                "sl_optimizers": {
                    pid: l.sl_optimizer.state_dict() for pid, l in self.learners.items()
                },
            }
        super()._save_checkpoint(checkpoint_data)

    def load_checkpoint_weights(self, checkpoint: Dict[str, Any]):
        """Loads NFSP weights."""
        shared = checkpoint.get("shared_networks", True)
        if shared:
            if "br_agent_network" in checkpoint:
                self.br_agent_network.load_state_dict(checkpoint["br_agent_network"])
            if "avg_agent_network" in checkpoint:
                self.avg_agent_network.load_state_dict(checkpoint["avg_agent_network"])
            if "rl_optimizer" in checkpoint:
                self.learner.rl_learner.optimizer.load_state_dict(
                    checkpoint["rl_optimizer"]
                )
            if "sl_optimizer" in checkpoint:
                self.learner.sl_optimizer.load_state_dict(checkpoint["sl_optimizer"])
        else:
            if "br_agent_networks" in checkpoint:
                for pid, sd in checkpoint["br_agent_networks"].items():
                    if pid in self.br_agent_networks:
                        self.br_agent_networks[pid].load_state_dict(sd)
            if "avg_agent_networks" in checkpoint:
                for pid, sd in checkpoint["avg_agent_networks"].items():
                    if pid in self.avg_agent_networks:
                        self.avg_agent_networks[pid].load_state_dict(sd)
            if "rl_optimizers" in checkpoint:
                for pid, sd in checkpoint["rl_optimizers"].items():
                    if pid in self.learners:
                        self.learners[pid].rl_learner.optimizer.load_state_dict(sd)
            if "sl_optimizers" in checkpoint:
                for pid, sd in checkpoint["sl_optimizers"].items():
                    if pid in self.learners:
                        self.learners[pid].sl_optimizer.load_state_dict(sd)

    def select_test_action(self, state, info, env) -> Any:
        """Select action for testing (from average strategy)."""
        # eta=0 for average strategy testing
        return self.policy.compute_action(state, info, eta=0.0).item()

    def _setup_stats(self) -> None:
        super()._setup_stats()
        # Additional NFSP specific keys
        stat_keys = ["rl_loss", "sl_loss", "sl_policy"]
        for key in stat_keys:
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
