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
        recording_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        replay_buffer: Optional[Any] = None,
    ):
        self.interact_graph = interact_graph
        self.env = env
        self.recording_fn = recording_fn
        self.replay_buffer = replay_buffer
        self.current_obs = None
        self.episode_return = 0.0

    def reset(self, context: Optional[ExecutionContext] = None) -> torch.Tensor:
        if context and context.episode_step > 0:
            print(
                f"[Actor] Episode {context.episode_count} finished. Return: {self.episode_return:.2f} | Steps: {context.episode_step}"
            )

        obs, _ = self.env.reset()
        self.current_obs = torch.tensor(obs, dtype=torch.float32)
        self.episode_return = 0.0
        if context:
            context.episode_count += 1
            context.episode_step = 0
        return self.current_obs

    def step(self, context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """Performs a single environment step and returns the trace."""
        context = context or ExecutionContext()
        if self.current_obs is None:
            self.reset(context=context)

        # Bind snapshots for all actors defined in the graph that have parameters
        for nid, node in self.interact_graph.nodes.items():
            if "param_store" in node.params:
                ps = node.params["param_store"]
                if isinstance(ps, ParameterStore):
                    if not context.get_actor_snapshot(nid):
                        snapshot = ActorSnapshot(ps.version, ps.get_state())
                        context.bind_actor(nid, snapshot)

        results = execute(
            self.interact_graph,
            initial_inputs={
                "obs_in": self.current_obs,
                "clock_in": torch.tensor(context.env_step, dtype=torch.int64)
            },
            context=context,
        )
        action_res = results.get("actor")
        # Unwrap if it's a RuntimeValue (Value object)
        from runtime.values import Value

        # TODO: this shouldnt really be necessary, and the fact that it is, is concerning. See if these kinds of contracts can be improved
        action_data = action_res.data if isinstance(action_res, Value) else action_res

        # TODO: this is also not great, the actor node should really be returning the action directly. or at least these isinstance checks should be unecessary
        if isinstance(action_data, dict):
            action = action_data["action"]
        else:
            action = action_data

        next_obs_raw, reward, terminated, truncated, _ = self.env.step(action)
        self.episode_return += float(reward)
        done = terminated or truncated
        next_obs = torch.tensor(next_obs_raw, dtype=torch.float32)

        step_data = {
            "obs": self.current_obs,
            "action": action,
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_obs": next_obs,
            "done": torch.tensor(float(done)),
            "metadata": {
                "step_index": context.actor_step,
                "episode_id": context.episode_count,
                "episode_step": context.episode_step,
                "actor_results": results,
                "context": context.to_dict(),
            },
        }

        # Handle recording
        # TODO: what is this? what does recording function do? i dont love defaults and branching like this, maybe something to work on
        if self.recording_fn:
            self.recording_fn(step_data)
        elif self.replay_buffer is not None:
            # Default behavior: add to buffer
            self.replay_buffer.add(step_data)

        self.current_obs = next_obs
        context.actor_step += 1
        context.env_step += 1
        context.episode_step += 1
        context.global_step += 1

        if done:
            self.reset(context=context)
        return step_data

    def collect_trajectory(
        self, max_steps: int, context: Optional[ExecutionContext] = None
    ) -> List[Dict[str, Any]]:
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

    def __init__(self, train_graph: Graph, replay_buffer: Optional[Any] = None):
        self.train_graph = train_graph
        self.replay_buffer = replay_buffer

    def update_step(
        self,
        batch: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """
        Performs a single optimization step.
        If batch is not provided, samples from the internal buffer.
        """
        context = context or ExecutionContext()

        initial_inputs = {}
        if batch is not None:
            initial_inputs["traj_in"] = batch

        results = execute(
            self.train_graph, initial_inputs=initial_inputs, context=context
        )
        context.learner_step += 1
        
        # Unwrap values for usability (keep Skipped/NoOp as is)
        unwrapped = {}
        for k, v in results.items():
            if hasattr(v, "has_data") and v.has_data:
                unwrapped[k] = v.data
            else:
                unwrapped[k] = v
        return unwrapped

    def execute(
        self,
        batch: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """Synonym for update_step."""
        return self.update_step(batch, context)
