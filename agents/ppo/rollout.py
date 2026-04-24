import torch
from typing import Dict, Any, Callable
from runtime.state import ReplayBuffer


def create_ppo_recording_fn(buffer: ReplayBuffer) -> Callable[[Dict[str, Any]], None]:
    """
    Create a recording function for PPO that flattens actor results.

    Args:
        buffer: The replay buffer to add data to.

    Returns:
        A function that takes step_data and adds it to the buffer.
    """

    def ppo_record(step_data: Dict[str, Any]) -> None:
        # Flatten actor_results into top level for the collator
        metadata = step_data.get("metadata", {})
        results = metadata.get("actor_results", {})
        actor_data = results.get("actor")
        env_idx = metadata.get("env_idx")
        # Extract directly from actor_data (already unwrapped by ActorRuntime)
        buffer.add(
            obs=step_data["obs"],
            action=step_data["action"],
            reward=step_data["reward"],
            terminated=step_data["terminated"],
            truncated=step_data["truncated"],
            value=actor_data["value"],
            log_prob=actor_data["log_prob"],
            policy_version=actor_data["policy_version"],
            env_idx=env_idx,
        )

    return ppo_record
