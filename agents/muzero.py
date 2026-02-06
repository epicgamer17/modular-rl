import copy
import datetime
import random
import sys

from replay_buffers.buffer_factories import create_muzero_buffer
from replay_buffers.game import Game, TimeStep
from search.search_factories import create_mcts


sys.path.append("../")
from time import time
import traceback
from modules.utils import scalar_to_support, support_to_scalar, get_lr_scheduler
import numpy as np
from stats.stats import PlotType, StatTracker
from losses.losses import create_muzero_loss_pipeline

from agents.agent import MARLBaseAgent
from agent_configs.muzero_config import MuZeroConfig
import torch
import torch.nn.functional as F
from modules.agent_nets.muzero import Network
import datetime

from replay_buffers.utils import update_per_beta

from modules.utils import scale_gradient

from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp
from tqdm import tqdm

from agents.torch_mp_agent import TorchMPAgent
from agents.muzero_actor import MuZeroActor
from agents.muzero_learner import MuZeroLearner


class MuZeroAgent(MARLBaseAgent, TorchMPAgent):
    def __init__(
        self,
        env,
        config: MuZeroConfig,
        name=datetime.datetime.now().timestamp(),
        test_agents=[],
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            else (
                torch.device("mps")
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else torch.device("cpu")
            )
        ),
        from_checkpoint=False,
        loss_pipeline=None,
    ):
        super(MuZeroAgent, self).__init__(
            env,
            config,
            name,
            test_agents=test_agents,
            device=device,
            from_checkpoint=from_checkpoint,
        )
        self.env.reset()  # for multiprocessing

        # Add learning rate scheduler
        self.model = Network(
            config=config,
            num_actions=self.num_actions,
            input_shape=torch.Size(
                (self.config.minibatch_size,) + self.observation_dimensions
            ),
            # TODO: sort out when to do channel first and channel last
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
        )

        self.actor = MuZeroActor(
            self.config, self.device, self.num_actions, self.observation_dimensions
        )

        self.learner = MuZeroLearner(
            config=self.config,
            model=self.model,
            device=self.device,
            num_actions=self.num_actions,
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.observation_dtype,
            predict_initial_inference_fn=self.actor.predict_initial_inference,
            predict_recurrent_inference_fn=self.actor.predict_recurrent_inference,
            predict_afterstate_recurrent_inference_fn=self.actor.predict_afterstate_recurrent_inference,
            preprocess_fn=self.actor.preprocess,
        )

        if self.config.multi_process:
            self.model.share_memory()

        self.target_model = Network(
            config=self.config,
            num_actions=self.num_actions,
            input_shape=torch.Size(
                (self.config.minibatch_size,) + self.observation_dimensions
            ),
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
        )
        # copy weights
        self.target_model.load_state_dict(self.model.state_dict())

        if self.config.multi_process:
            self.target_model.share_memory()
        else:
            self.target_model.to(self.device)

        if not self.config.multi_process:
            self.model.to(self.device)

        # Expose components for backward compatibility
        self.replay_buffer = self.learner.replay_buffer
        self.optimizer = self.learner.optimizer
        self.lr_scheduler = self.learner.lr_scheduler
        self.loss_pipeline = self.learner.loss_pipeline

        test_score_keys = [
            "test_score_vs_{}".format(agent.model_name) for agent in self.test_agents
        ]
        self._setup_stats()
        if self.config.multi_process:
            self.setup_mp()
        else:

            class StopFlag:
                def __init__(self, val):
                    self.value = val

            self.stop_flag = StopFlag(0)
        self.testing_worker = None

    def _setup_stats(self):
        """Initializes or updates the stat tracker with all required keys and plot types."""
        test_score_keys = [
            "test_score_vs_{}".format(agent.model_name) for agent in self.test_agents
        ]

        # EnsureStatTracker exists (might have been deleted in __getstate__)
        if not hasattr(self, "stats") or self.stats is None:
            self.stats = StatTracker(model_name=self.model_name)

        # 1. Initialize Keys (if not already present)
        stat_keys = [
            "score",
            "policy_loss",
            "value_loss",
            "reward_loss",
            "to_play_loss",
            "cons_loss",
            "loss",
            "test_score",
            "episode_length",
            "policy_entropy",
            "value_diff",
            "policy_improvement",
        ] + test_score_keys

        if self.config.stochastic:
            stat_keys += [
                "num_codes",
                "chance_probs",
                "chance_entropy",
                "q_loss",
                "sigma_loss",
                "vqvae_commitment_cost",
            ]

        target_values = {
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
            "num_codes": 1 if self.config.game.is_deterministic else None,
        }

        use_tensor_dicts = {
            "test_score": ["score", "max_score", "min_score"],
            "policy_improvement": ["network", "search"],
            **{
                key: ["score"]
                + [
                    "player_{}_score".format(p)
                    for p in range(self.config.game.num_players)
                ]
                + [
                    "player_{}_win%".format(p)
                    for p in range(self.config.game.num_players)
                ]
                for key in test_score_keys
            },
        }

        # For host StatTracker, initialize keys that don't exist
        if not self.stats._is_client:
            for key in stat_keys:
                if key not in self.stats.stats:
                    self.stats._init_key(
                        key,
                        target_value=target_values.get(key),
                        subkeys=use_tensor_dicts.get(key),
                    )

        # 2. Add/Refresh Plot Types
        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
            PlotType.BEST_FIT_LINE,
            rolling_window=100,
            ema_beta=0.6,
        )
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types(
            "policy_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "reward_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "to_play_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("cons_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("num_codes", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("q_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "sigma_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "vqvae_commitment_cost", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "policy_entropy", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_diff", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("policy_improvement", PlotType.BAR)

        if self.config.stochastic:
            self.stats.add_plot_types("chance_probs", PlotType.BAR)
            self.stats.add_plot_types(
                "chance_entropy", PlotType.ROLLING_AVG, rolling_window=100
            )

    def worker_fn(
        self, worker_id, stop_flag, stats_client: StatTracker, error_queue: mp.Queue
    ):
        self.stats = stats_client
        print(f"[Worker {worker_id}] Starting self-play...")
        worker_env = self.config.game.make_env()  # each process needs its own env
        try:
            from wrappers import record_video_wrapper

            worker_env.render_mode = "rgb_array"
            worker_env = record_video_wrapper(
                worker_env,
                f"./videos/{self.model_name}/{worker_id}",
                self.checkpoint_interval,
            )
        except:
            print(f"[Worker {worker_id}] Could not record video")
        # Workers should use the target model for inference so training doesn't
        # destabilize ongoing self-play. Ensure the target model is on the worker's device
        # and set as the inference model.
        self.target_model.to(self.device)
        self.target_model.eval()

        try:
            while not stop_flag.value:
                if (
                    random.random() < self.config.reanalyze_ratio
                    and self.replay_buffer.size > 0
                ):
                    self.reanalyze_game(inference_model=self.target_model)
                else:
                    score, num_steps = self.play_game(
                        env=worker_env, inference_model=self.target_model
                    )
                    # print(f"[Worker {worker_id}] Finished a game with score {score}")
                    # worker_env.close()  # for saving video
                    stats_client.append("score", score)
                    stats_client.append("episode_length", num_steps)
                    stats_client.increment_steps(num_steps)
        except Exception as e:
            # Send both exception and traceback back
            error_queue.put((e, traceback.format_exc()))
            raise  # ensures worker process exits with error
        worker_env.close()

    def train(self):
        self._setup_stats()
        if self.config.multi_process:
            stats_client = self.stats.get_client()
            self.start_workers(
                worker_fn=self.worker_fn,
                num_workers=self.config.num_workers,
                stats_client=stats_client,
            )

        start_time = time() - self.stats.get_time_elapsed()
        self.model.to(self.device)

        # ensure inference uses the current target before any play in main thread
        self.inference_model = self.target_model

        while self.training_step < self.config.training_steps:
            if self.config.multi_process:
                self.check_worker_errors()
                self.stats.drain_queue()
            if not self.config.multi_process:
                for training_game in tqdm(range(self.config.games_per_generation)):
                    if self.stop_flag.value:
                        print("Stopping game generation")
                        break

                    score, num_steps = self.play_game(inference_model=self.target_model)
                    self.stats.append("score", score)
                    self.stats.increment_steps(num_steps)
                if self.stop_flag.value:
                    print("Stopping training")
                    break

            # STAT TRACKING
            if (self.training_step * self.config.minibatch_size + 1) / (
                max(0, self.stats.get_num_steps() - self.config.min_replay_buffer_size)
                + 1
            ) > self.config.lr_ratio:
                continue

            if self.learner.replay_buffer.size >= self.config.min_replay_buffer_size:
                for minibatch in range(self.config.num_minibatches):
                    loss_stats = self.learner.step(self.stats)
                    if loss_stats:
                        for key, val in loss_stats.items():
                            self.stats.append(key, val)
                self.training_step += 1

                self.learner.replay_buffer.set_beta(
                    update_per_beta(
                        self.learner.replay_buffer.beta,
                        self.config.per_beta_final,
                        self.config.training_steps,
                        self.config.per_beta,
                    )
                )

                if self.training_step % self.config.transfer_interval == 0:
                    self.update_target_model()

            if self.training_step % self.test_interval == 0 and self.training_step > 0:
                if self.config.multi_process:
                    try:
                        # Check if previous testing worker is still running
                        if (
                            self.testing_worker is None
                            or not self.testing_worker.is_alive()
                        ):
                            self.testing_worker = mp.Process(
                                target=self.run_tests, args=(stats_client,)
                            )
                            self.testing_worker.start()
                        self.stats.drain_queue()
                    except Exception as e:
                        print(f"Error starting testing worker: {e}")
                else:
                    self.run_tests(stats=self.stats)

            # CHECKPOINTING
            if (
                self.training_step % self.checkpoint_interval == 0
                and self.training_step > 0
            ):
                self.stats.set_time_elapsed(time() - start_time)
                # print("Saving Checkpoint")
                self.save_checkpoint(
                    save_weights=self.config.save_intermediate_weights,
                )
        if self.config.multi_process:
            self.stop_workers()
            print("All workers stopped")

        if self.config.multi_process:
            try:
                if self.testing_worker is not None:
                    self.testing_worker.join()
            except:
                pass
            self.stats.drain_queue()

        self.stats.set_time_elapsed(time() - start_time)
        print("Finished Training")
        self.run_tests(self.stats)
        self.save_checkpoint(save_weights=True)
        self.env.close()

    def learn(self):
        loss_stats = self.learner.step(self.stats)
        if loss_stats:
            return (
                loss_stats["value_loss"],
                loss_stats["policy_loss"],
                loss_stats["reward_loss"],
                loss_stats["to_play_loss"],
                loss_stats["cons_loss"],
                loss_stats["q_loss"],
                loss_stats["sigma_loss"],
                loss_stats["vqvae_commitment_cost"],
                loss_stats["loss"],
            )
        return None

    def predict(
        self,
        state,
        info: dict = None,
        env=None,
        inference_model=None,
        *args,
        **kwargs,
    ):
        if inference_model is None:
            inference_model = self.model
        return self.actor.predict(state, info, env=env, inference_model=inference_model)

    def select_actions(
        self,
        prediction,
        temperature=0.0,
        *args,
        **kwargs,
    ):
        return self.actor.select_actions(prediction, temperature=temperature)

    def play_game(self, env=None, inference_model=None):
        if env is None:
            env = self.env

        # Use actor to play the game
        game = self.actor.play_game(
            env=env,
            model=inference_model if inference_model is not None else self.model,
            stats_tracker=self.stats,
        )

        # Store in replay buffer
        self.replay_buffer.store_aggregate(game_object=game)

        if self.config.game.num_players != 1:
            return env.rewards[env.agents[0]], len(game)
        else:
            return sum(game.rewards), len(game)

    def reanalyze_game(self, inference_model=None):
        if inference_model is None:
            inference_model = self.model
        # or reanalyze buffer
        with torch.no_grad():
            sample = self.replay_buffer.sample_game()
            observations = sample["observations"]
            root_values = sample["values"].to(self.device)[:, 0]
            policies = sample["policies"].to(self.device)[:, 0]
            traj_actions = sample["actions"].to(self.device)[:, 0]
            traj_to_plays = sample["to_plays"].to(self.device)[:, 0]
            legal_moves_masks = sample["legal_moves_masks"].to(self.device)
            indices = sample["indices"]
            ids = sample["ids"]

            new_policies = []
            new_root_values = []
            new_priorities = []
            infos = []
            for (
                idx,
                obs,
                root_value,
                traj_action,
                traj_to_play,
                mask,
            ) in zip(
                indices,
                observations,
                root_values,
                traj_actions,
                traj_to_plays,
                legal_moves_masks,
            ):
                to_play = int(torch.argmax(traj_to_play).item())
                if self.config.game.has_legal_moves:
                    info = {
                        "legal_moves": torch.nonzero(mask).view(-1).tolist(),
                        "player": to_play,
                    }
                else:
                    info = {
                        "player": to_play,
                    }

                if not (
                    self.config.game.has_legal_moves and len(info["legal_moves"]) == 0
                ):
                    infos.append(info)
                    # print("info with legal moves from nonzero mask", info)
                    # ADD INJECTING SEEN ACTION THING FROM MUZERO UNPLUGGED
                    if self.config.reanalyze_method == "mcts":
                        root_value, _, new_policy, best_action, _ = (
                            self.actor.search.run(
                                obs,
                                info,
                                to_play,
                                {
                                    "initial": self.actor.predict_initial_inference,
                                    "recurrent": self.actor.predict_recurrent_inference,
                                    "afterstate": self.actor.predict_afterstate_recurrent_inference,
                                },
                                trajectory_action=int(traj_action.item()),
                                inference_model=inference_model,
                            )
                        )

                        new_root_value = float(root_value)
                    else:
                        value, new_policy, _ = self.actor.predict_initial_inference(
                            obs, model=inference_model
                        )
                        new_root_value = value.item()
                else:
                    infos.append(info)
                    new_policy = torch.ones_like(policies[0]) / self.num_actions
                    new_root_value = 0.0

                new_policies.append(new_policy)
                new_root_values.append(new_root_value)

                # decide value target per your config (paper default: keep stored n-step TD for Atari)
            # now write back under write_lock and update priorities with ids
            self.replay_buffer.reanalyze_game(
                indices,
                new_policies,
                new_root_values,
                ids,
                self.training_step,
                self.config.training_steps,
            )
            if self.config.reanalyze_update_priorities:
                stored_n_step_value = float(
                    self.replay_buffer.n_step_values_buffer[idx][0].item()
                )

                new_policies.append(new_policy[0])
                new_root_values.append(new_root_value)
                new_priorities.append(abs(float(root_value) - stored_n_step_value))

                self.update_replay_priorities(
                    indices, new_priorities, ids=np.array(ids)
                )

    def _track_search_stats(self, search_metadata):
        """Track statistics from the search process."""
        if search_metadata is None:
            return

        network_policy = search_metadata["network_policy"]
        search_policy = search_metadata["search_policy"]
        network_value = search_metadata["network_value"]
        search_value = search_metadata["search_value"]

        # 1. Policy Entropy
        # search_policy: (num_actions,)
        probs = search_policy + 1e-10
        entropy = -torch.sum(probs * torch.log(probs)).item()
        self.stats.append("policy_entropy", entropy)

        # 2. Value Difference
        self.stats.append("value_diff", abs(search_value - network_value))

        # 3. Policy Improvement (BAR plot comparison)
        self.stats.append(
            "policy_improvement", network_policy.unsqueeze(0), subkey="network"
        )
        self.stats.append(
            "policy_improvement", search_policy.unsqueeze(0), subkey="search"
        )

    def __getstate__(self):
        # 1. Start with a copy of state from parents/mixins
        try:
            state = super().__getstate__()
            state = state.copy()
        except AttributeError:
            state = self.__dict__.copy()

        # 2. Exclude multiprocessing specific objects or large learner objects

        # ALWAYS exclude these - they are only needed by the learner/main process
        if "optimizer" in state:
            del state["optimizer"]
        if "lr_scheduler" in state:
            del state["lr_scheduler"]
        if "loss_pipeline" in state:
            del state["loss_pipeline"]

        # If the user has used torch.compile on the model, it might contain unpicklable weakrefs.
        if "model" in state and hasattr(self.model, "_orig_mod"):
            state["model_state_dict"] = {
                k: v.cpu() for k, v in self.model.state_dict().items()
            }
            del state["model"]

        if "target_model" in state and hasattr(self.target_model, "_orig_mod"):
            del state["target_model"]

        # Only handle these if training has started (step > 0)
        if self.training_step > 0:
            if "model" in state:
                state["model_state_dict"] = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }
                del state["model"]

            if "target_model" in state:
                del state["target_model"]

            if "replay_buffer" in state:
                del state["replay_buffer"]

        return state

    def __setstate__(self, state):
        model_state_dict = state.pop("model_state_dict", None)
        self.__dict__.update(state)
        # self.stop_flag is already in state if we didn't convert it to int

        # env and test_env were deleted in __getstate__, re-initialize
        self.env = self.config.game.make_env()
        self.test_env = self.config.game.make_env(render_mode="rgb_array")

        # Reconstruct model if we have weights (usually for step > 0)
        if model_state_dict is not None:
            # self.config, self.observation_dimensions, self.num_actions are in state
            device = torch.device("cpu")  # Initialize on CPU
            self.model = Network(
                config=self.config,
                num_actions=self.num_actions,
                input_shape=torch.Size(
                    (self.config.minibatch_size,) + self.observation_dimensions
                ),
                channel_first=True,
                world_model_cls=self.config.world_model_cls,
            )
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.target_model = copy.deepcopy(self.model)
            # Move target to shared memory if multi_process (though for testing we might just use local copy)
            if self.config.multi_process:
                # Note: sharing CUDA/MPS tensors is tricky. If device is CPU, this is fine.
                # If device is MPS, this might fail or be no-op.
                # Given we just deserialized, we are likely fine keeping it local for the test worker.
                # But if this is a training worker, it might expect shared memory?
                # Training workers use target_model for inference.
                # A reconstructed target_model here is NOT shared with the main process 'target_model'.
                # This implies __setstate__ is creating a LOCAL copy.
                # If training workers need the SHARED target model updated by learner,
                # then training workers CANNOT rely on this reconstruction!
                # Training workers rely on the pickled 'target_model' which points to shared memory.
                # WE DELETED 'target_model' from state!
                pass
