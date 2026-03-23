from typing import Dict, Any, Optional
from replay_buffers.sequence import Sequence

class SequenceManager:
    """
    Utility to handle the chunking and flushing logic of RL trajectories.
    Normally owned by Actors/Workers to manage multiple vectorized environments.
    """

    def __init__(self, num_players: int, num_envs: int):
        """
        Initializes the SequenceManager.

        Args:
            num_players: Number of players in the environment.
            num_envs: Number of concurrent environments to manage.
        """
        self.num_players = num_players
        self.num_envs = num_envs
        self.active_sequences: Dict[int, Sequence] = {
            i: Sequence(num_players) for i in range(num_envs)
        }

    def append(self, env_id: int, transition: Dict[str, Any]) -> None:
        """
        Appends a transition to the sequence associated with a specific env_id.

        Args:
            env_id: The ID of the environment (0 to num_envs-1).
            transition: Dictionary containing transition data.
        """
        assert env_id in self.active_sequences, f"Invalid env_id {env_id}"
        self.active_sequences[env_id].append(**transition)

    def flush(self, env_id: int) -> Sequence:
        """
        Returns the completed Sequence object for the given env_id and
        immediately resets the slot with a fresh Sequence.

        Args:
            env_id: The ID of the environment to flush.

        Returns:
            The completed Sequence object.
        """
        assert env_id in self.active_sequences, f"Invalid env_id {env_id}"
        completed_sequence = self.active_sequences[env_id]
        self.active_sequences[env_id] = Sequence(self.num_players)
        return completed_sequence

    def get_sequence(self, env_id: int) -> Sequence:
        """
        Retrieves the active sequence for a specific env_id without flushing it.

        Args:
            env_id: The ID of the environment.

        Returns:
            The active Sequence object.
        """
        return self.active_sequences[env_id]
