"""
RolloutController for orchestrating the interaction between actors and environments.
Handles metadata attachment and recording to buffers.
"""

from typing import Any, Dict, List, Optional, Callable
import torch
from core.graph import Graph
from runtime.executor import execute
from runtime.state import ReplayBuffer

class RolloutController:
    """
    Orchestrates the Rollout loop: Actor -> Environment -> Storage.
    """
    def __init__(
        self, 
        interact_graph: Graph, 
        env: Any,
        recording_fn: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.interact_graph = interact_graph
        self.env = env
        self.recording_fn = recording_fn
        self.current_obs = None
        self._step_count = 0
        self._episode_count = 0

    def reset(self) -> torch.Tensor:
        """Resets the environment and internal counters."""
        obs, _ = self.env.reset()
        self.current_obs = torch.tensor(obs, dtype=torch.float32)
        self._episode_count += 1
        return self.current_obs

    def rollout_step(self) -> Dict[str, Any]:
        """
        Performs a single rollout step:
        1. Invoke Actor via Graph
        2. Step Environment
        3. Attach Metadata
        4. Record to buffers
        """
        if self.current_obs is None:
            self.reset()
            
        # 1. Invoke Actor
        results = execute(self.interact_graph, initial_inputs={"obs_in": self.current_obs})
        # We assume the graph has an 'actor' node or similar. 
        # For generality, we can take the last node in topological order or a specific ID.
        # Here we look for a node with 'action' in its output or just use 'actor' ID.
        action_data = results.get("actor")
        
        # Handle cases where actor output is a dict (like PPO) or a raw value
        if isinstance(action_data, dict):
            action = action_data["action"]
        else:
            action = action_data

        # 2. Step Environment
        next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        next_obs = torch.tensor(next_obs_raw, dtype=torch.float32)

        # 3. Build Trace & Attach Metadata
        step_data = {
            "obs": self.current_obs,
            "action": action,
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": next_obs,
            "done": torch.tensor(float(done)),
            "metadata": {
                "step_index": self._step_count,
                "episode_id": self._episode_count,
                "actor_results": results
            }
        }

        # 4. Record
        if self.recording_fn:
            self.recording_fn(step_data)

        self.current_obs = next_obs
        self._step_count += 1
        
        if done:
            self.reset()
            
        return step_data

    def collect_trajectory(self, max_steps: int) -> List[Dict[str, Any]]:
        """Collects a sequence of steps until done or max_steps is reached."""
        trajectory = []
        for _ in range(max_steps):
            step_data = self.rollout_step()
            trajectory.append(step_data)
            if step_data["done"].item() > 0:
                break
        return trajectory
