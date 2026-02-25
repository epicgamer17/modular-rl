"""
PufferActor Module

This module defines the PufferActor, which is a 'Fat Worker' that runs PufferLib
and pivots the data from horizontal batches back into vertical, chronological Sequence objects.
"""

import torch
import torch.multiprocessing as mp
import pufferlib.vector
from replay_buffers.sequence import Sequence


class PufferActor(mp.Process):
    """
    The Active Episode Tracker that runs PufferLib inside a dedicated multiprocessing Process.

    This actor leverages PufferLib for vectorized environment execution and bridges
    the horizontal batched outputs from PufferLib back into chronological vertical
    Sequence objects for RL training.
    """

    def __init__(self, config, env_creator, shared_network, shared_buffer, search_alg):
        """
        Initializes the PufferActor.

        Args:
            config: The configuration object for the agent/worker.
            env_creator: A callable that returns an instance of the environment.
            shared_network: The neural network shared across processes.
            shared_buffer: The multiprocessing-safe shared replay buffer.
            search_alg: The search algorithm (e.g., MCTS) used to select actions.
        """
        super().__init__()
        self.config = config
        self.num_envs = config.num_envs_per_worker
        self.shared_network = shared_network
        self.shared_buffer = shared_buffer
        self.search_alg = search_alg

        # Instantiate PufferLib INSIDE the process
        self.vec_env = pufferlib.vector.Multiprocessing(
            env_creator=env_creator, num_envs=self.num_envs, num_workers=2
        )
        self.active_sequences = [
            Sequence(config.num_players) for _ in range(self.num_envs)
        ]

    def run(self):
        """
        Main execution loop for the PufferActor.

        This loop continuously steps the vectorized environments, runs batched MCTS
        to select actions, and pivots the batched data into individual episode sequences.
        When an episode finishes, the sequence is packed and stored in the shared replay buffer.
        """
        # Configure thread affinity to avoid OpenMP contention
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        self.vec_env.async_reset()
        obs, _, _, _, infos, _, _ = self.vec_env.recv()

        while True:
            # 1. Batched MCTS
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            to_play = [info.get("player", 0) for info in infos]

            with torch.inference_mode():
                actions, policies, values = self.search_alg.run_vectorized(
                    obs_tensor, infos, to_play, self.shared_network
                )

            # 2. Step Environments
            self.vec_env.send(actions)
            next_obs, rewards, terminals, truncs, next_infos, _, _ = self.vec_env.recv()

            # 3. Pivot Batch to Chronological Sequences
            for i in range(self.num_envs):
                self.active_sequences[i].append(
                    observation=obs[i],
                    action=actions[i],
                    reward=rewards[i],
                    policy=policies[i],
                    value=values[i],
                    terminated=terminals[i],
                    truncated=truncs[i],
                    player_id=to_play[i],
                    legal_moves=infos[i].get("legal_moves", []),
                )

                # 4. Handle End of Episode
                if terminals[i] or truncs[i]:
                    completed_seq = self.active_sequences[i]

                    # Store safely in the locked shared buffer
                    self.shared_buffer.store_aggregate(completed_seq)
                    self.active_sequences[i] = Sequence(self.config.num_players)

            obs, infos = next_obs, next_infos
