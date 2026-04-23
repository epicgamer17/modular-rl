"""
Split Runtimes for Actor (Online) and Learner (Offline) execution.
Decouples data collection from model optimization.
"""

from typing import Any, Dict, List, Optional, Callable
import torch
from core.graph import Graph
from runtime.executor import execute
from runtime.context import ExecutionContext, ActorSnapshot
from runtime.state import ParameterStore

class ActorRuntime:
    """
    Online system responsible for environment interaction and trace generation.
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
        obs, _ = self.env.reset()
        self.current_obs = torch.tensor(obs, dtype=torch.float32)
        self._episode_count += 1
        return self.current_obs

    def step(self, context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """Performs a single environment step and returns the trace."""
        context = context or ExecutionContext()
        if self.current_obs is None:
            self.reset()
            
        # Bind snapshots for all actors defined in the graph that have parameters
        for nid, node in self.interact_graph.nodes.items():
            if "param_store" in node.params:
                ps = node.params["param_store"]
                if isinstance(ps, ParameterStore):
                    if not context.get_actor_snapshot(nid):
                        snapshot = ActorSnapshot(ps.version, ps.get_parameters())
                        context.bind_actor(nid, snapshot)

        results = execute(self.interact_graph, initial_inputs={"obs_in": self.current_obs}, context=context)
        action_data = results.get("actor")
        
        if isinstance(action_data, dict):
            action = action_data["action"]
        else:
            action = action_data

        next_obs_raw, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_obs = torch.tensor(next_obs_raw, dtype=torch.float32)

        step_data = {
            "obs": self.current_obs,
            "action": action,
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": next_obs,
            "done": torch.tensor(float(done)),
            "metadata": {
                "step_index": self._step_count,
                "episode_id": self._episode_count,
                "actor_results": results,
                "context": context.to_dict()
            }
        }

        if self.recording_fn:
            self.recording_fn(step_data)

        self.current_obs = next_obs
        self._step_count += 1
        if done: self.reset()
        return step_data

    def collect_trajectory(self, max_steps: int, context: Optional[ExecutionContext] = None) -> List[Dict[str, Any]]:
        """Collects a sequence of steps until done or max_steps is reached."""
        trajectory = []
        for _ in range(max_steps):
            step_data = self.step(context)
            trajectory.append(step_data)
            if step_data["done"].item() > 0:
                break
        return trajectory

class LearnerRuntime:
    """
    Offline system responsible for sampling data and updating parameters.
    """
    def __init__(
        self,
        train_graph: Graph,
        buffer: Optional[Any] = None
    ):
        self.train_graph = train_graph
        self.buffer = buffer
        self._update_count = 0

    def update_step(self, batch: Optional[Dict[str, Any]] = None, context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """
        Performs a single optimization step.
        If batch is not provided, samples from the internal buffer.
        """
        context = context or ExecutionContext()
        
        initial_inputs = {}
        if batch is not None:
            initial_inputs["traj_in"] = batch
        
        results = execute(self.train_graph, initial_inputs=initial_inputs, context=context)
        self._update_count += 1
        return results
