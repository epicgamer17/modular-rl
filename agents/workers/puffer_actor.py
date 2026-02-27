"""
PufferActor Module

This module defines the PufferActor, which is a 'Fat Worker' that runs PufferLib
and pivots the data from horizontal batches back into vertical, chronological Sequence objects.
"""

import torch
from typing import Callable, Any, Optional, Tuple, Dict

import pufferlib.vector
from replay_buffers.sequence import Sequence
from agents.workers.actors import BaseActor


def _make_puffer_env(env_factory: Callable[[], Any], buf: Any = None, seed: Any = None):
    import pufferlib.emulation

    env = pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=env_factory, buf=buf, seed=seed
    )

    # Note: PufferLib Multiprocessing skips sending info if it's empty/False.
    # This causes a desync between obs and infos in the batch.
    # We ensure info is always truthy.
    original_step = env.step

    def patched_step(action):
        obs, reward, term, trunc, info = original_step(action)
        if not info:
            info["_puffer_keep"] = True
        return obs, reward, term, trunc, info

    env.step = patched_step

    original_reset = env.reset

    def patched_reset(**kwargs):
        obs, info = original_reset(**kwargs)
        if not info:
            info["_puffer_keep"] = True
        return obs, info

    env.reset = patched_reset

    return env


class PufferActor(BaseActor):
    """
    The Active Episode Tracker that runs PufferLib as a 'Fat Worker'.

    This actor leverages PufferLib for vectorized environment execution and bridges
    the horizontal batched outputs from PufferLib back into chronological vertical
    Sequence objects for RL training.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_network: Any,
        action_selector: Any,
        replay_buffer: Any,
        num_players: int,
        config: Any,
        device: Optional[torch.device] = None,
        name: str = "puffer_actor",
        *,
        worker_id: int = 0,
    ):
        """
        Initializes the PufferActor.
        """
        super().__init__(
            env_factory=env_factory,
            agent_network=agent_network,
            action_selector=action_selector,
            replay_buffer=replay_buffer,
            num_players=num_players,
            config=config,
            device=device,
            name=name,
            worker_id=worker_id,
        )

        self.num_envs = config.num_envs_per_worker
        self.vec_env = None
        self.active_sequences = None
        self._initialized = False

        # Renamed for consistency with BaseActor
        self._obs = None
        self._infos = None

    def _detect_num_players(self) -> int:
        return self.num_players

    def _get_player_id(self) -> Optional[str]:
        return None

    def _finalize_episode_info(self, sequence: Sequence) -> None:
        pass

    def _get_score(self, sequence: Sequence) -> float:
        return sum(sequence.rewards)

    def _reset_env(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Initializes the vectorized environment on the first call.
        """
        if not self._initialized:
            self.vec_env = pufferlib.vector.Multiprocessing(
                env_creators=[_make_puffer_env] * self.num_envs,
                env_args=[(self.env_factory,)] * self.num_envs,
                env_kwargs=[{}] * self.num_envs,
                num_envs=self.num_envs,
                num_workers=2,
            )
            self.active_sequences = [
                Sequence(self.num_players) for _ in range(self.num_envs)
            ]

            # Initial reset of the vectorized environment
            self.vec_env.async_reset()
            self._obs, _, _, _, self._infos, _, _ = self.vec_env.recv()
            self._initialized = True

        return self._obs, self._infos

    def _step_env(self, action: Any) -> Any:
        # Not typically used in vectorized mode, which overrides play_sequence
        raise NotImplementedError(
            "PufferActor uses vectorized execution; use play_sequence()"
        )

    @torch.inference_mode()
    def play_sequence(self, stats_tracker: Optional[Any] = None) -> Dict[str, Any]:
        """
        Runs the vectorized environments until at least one episode completes.
        This provides compatibility with LocalExecutor.

        Returns:
            The stats of the first completed episode in this batch.
        """
        if not self._initialized:
            self.reset()

        completed_stats = []

        while not completed_stats:
            # 1. Batched Action Selection
            batched_info = {
                "legal_moves": [info.get("legal_moves", []) for info in self._infos],
                "player": [info.get("player", 0) for info in self._infos],
            }

            actions_tensor, metadata = self.selector.select_action(
                agent_network=self.agent_network,
                obs=torch.as_tensor(self._obs, dtype=torch.float32, device=self.device),
                info=batched_info,
                exploration=True,
            )
            actions = actions_tensor.cpu().numpy()

            # 2. Step Environments
            self.vec_env.send(actions)
            next_obs, rewards, terminals, truncs, next_infos, _, _ = self.vec_env.recv()

            # 3. Pivot Batch to Chronological Sequences
            # Metadata keys (policy, value) are likely batched lists or tensors
            policies = metadata.get("policy")
            values = metadata.get("value")

            for i in range(self.num_envs):
                self.active_sequences[i].append(
                    observation=self._obs[i],
                    action=actions[i],
                    reward=rewards[i],
                    policy=policies[i] if policies is not None else None,
                    value=values[i] if values is not None else None,
                    terminated=terminals[i],
                    truncated=truncs[i],
                    player_id=batched_info["player"][i],
                    legal_moves=batched_info["legal_moves"][i],
                )

                # 4. Handle End of Episode
                if terminals[i] or truncs[i]:
                    self.active_sequences[i].append(
                        observation=next_obs[i],
                        terminated=terminals[i],
                        truncated=truncs[i],
                    )
                    completed_seq = self.active_sequences[i]

                    # Store safely in the locked shared buffer
                    self.replay_buffer.store_aggregate(completed_seq)

                    # Collect stats
                    ep_stats = {
                        "episode_length": len(completed_seq),
                        "score": self._get_score(completed_seq),
                    }
                    completed_stats.append(ep_stats)

                    # Log to tracker if provided
                    if stats_tracker:
                        score = ep_stats["score"]
                        length = ep_stats["episode_length"]
                        stats_tracker.append("score", score)
                        stats_tracker.append("episode_length", length)
                        stats_tracker.increment_steps(length)

                    # Reset sequence
                    self.active_sequences[i] = Sequence(self.num_players)

            self._obs, self._infos = next_obs, next_infos

        # Return the first completed episode's stats
        return completed_stats[0]
