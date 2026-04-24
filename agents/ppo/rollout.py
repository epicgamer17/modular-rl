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
        results = step_data["metadata"].get("actor_results", {})
        actor_val = results.get("actor")
        
        if (
            actor_val
            and hasattr(actor_val, "data")
            and isinstance(actor_val.data, dict)
        ):
            step_data.update(actor_val.data)
            # Extract from step_data and its 'actor' results
            actor_res = step_data.get("actor", {})
            buffer.add(
                obs=step_data["obs"],
                action=step_data["action"],
                reward=step_data["reward"],
                terminated=step_data["terminated"],
                truncated=step_data["truncated"],
                value=torch.tensor(actor_res.get("value", 0.0)),
                log_prob=torch.tensor(actor_res.get("log_prob", 0.0)),
                policy_version=actor_res.get("policy_version", 0)
            )
        
    return ppo_record
