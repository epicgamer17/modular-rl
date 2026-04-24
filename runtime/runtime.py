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
        from runtime.environment import wrap_env

        self.interact_graph = interact_graph
        self.env = wrap_env(env)
        self.recording_fn = recording_fn
        self.replay_buffer = replay_buffer
        self.current_obs = None
        self.last_obs = None
        self.last_done = False
        self.last_terminated = False

        self.num_envs = self.env.num_envs
        self.episode_returns = torch.zeros(self.num_envs)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long)

        self.last_episode_return = 0.0
        self.last_episode_length = 0

    def reset(
        self, seed: Optional[int] = None, context: Optional[ExecutionContext] = None
    ) -> torch.Tensor:
        obs = self.env.reset(seed=seed)
        # Ensure it's a tensor and has batch dimension
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        self.current_obs = obs
        self.episode_returns.fill_(0.0)
        self.episode_lengths.fill_(0)

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

        if context:
            context.actor_step += 1
            context.env_step += 1
            context.global_step += 1
            # episode_step is now per-env, handled via self.episode_lengths
            self.episode_lengths += 1
            context.episode_step = self.episode_lengths

        results = execute(
            self.interact_graph,
            initial_inputs={
                "obs_in": self.current_obs,
                "clock_in": torch.tensor(context.env_step, dtype=torch.int64),
                "episode_step_in": self.episode_lengths,
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

        step_res = self.env.step(action)

        from runtime.environment import validate_step_result

        validate_step_result(step_res, self.env.num_envs)

        # Update accounting
        self.episode_returns += step_res.reward

        done = step_res.terminated | step_res.truncated

        if done.any():
            for i in range(self.num_envs):
                if done[i]:
                    # Single env metrics (for simple logging/access)
                    self.last_episode_return = self.episode_returns[i].item()
                    self.last_episode_length = self.episode_lengths[i].item()

                    # Reset trackers for this specific environment
                    self.episode_returns[i] = 0.0
                    self.episode_lengths[i] = 0

                    if context:
                        context.episode_count += 1

        step_data = {
            "obs": self.current_obs,
            "action": action,
            "reward": step_res.reward,
            "next_obs": step_res.obs,
            "done": done.float(),
            "terminated": step_res.terminated.float(),
            "truncated": step_res.truncated.float(),
            "metadata": {
                "step_index": context.actor_step,
                "episode_id": context.episode_count,
                "episode_step": context.episode_step,
                "actor_results": results,
                "context": context.to_dict(),
            },
        }

        # 4. State Update
        # Update current observation, handling manual reset if necessary
        if done.any() and not self.env.auto_reset:
            # If any env is done and adapter doesn't auto-reset, we must manual reset
            # For now, we reset the whole adapter (standard gym vector reset behavior)
            # This is correct for B=1 and standard for many vector envs.
            # TODO: do we want to keep this? we could instead just reset the done environments, or is this handled internally in the vector env reset()?
            self.current_obs = self.env.reset()
        else:
            # Environment either auto-resetted or is not done
            self.current_obs = step_res.obs

        self.last_obs = step_res.obs
        self.last_done = done.any().item()
        self.last_terminated = step_res.terminated.any().item()

        # Handle recording
        # TODO: what is this? what does recording function do? i dont love defaults and branching like this, maybe something to work on
        if self.recording_fn:
            self.recording_fn(step_data)
        elif self.replay_buffer is not None:
            # ReplayBuffer expects individual transitions.
            # We unbatch the StepResult into individual transitions.
            batch_size = step_res.obs.shape[0]
            # TODO: are there problems with interleaving here? what if we need sequential data? does doing what we do here with vector envs cause problems?
            for i in range(batch_size):
                single_step = {}
                for k, v in step_data.items():
                    if k == "metadata":
                        # Unbatch metadata fields
                        single_metadata = {}
                        for mk, mv in v.items():
                            if (
                                isinstance(mv, torch.Tensor)
                                and mv.ndim > 0
                                and mv.shape[0] == batch_size
                            ):
                                single_metadata[mk] = mv[i]
                            elif isinstance(mv, dict):
                                # Recursively unbatch dictionaries (like actor_results)
                                single_subdict = {}
                                for sk, sv in mv.items():
                                    # Handle both Value objects and raw tensors
                                    from runtime.values import Value

                                    data = sv.data if isinstance(sv, Value) else sv
                                    if (
                                        isinstance(data, torch.Tensor)
                                        and data.ndim > 0
                                        and data.shape[0] == batch_size
                                    ):
                                        single_subdict[sk] = data[i]
                                    else:
                                        single_subdict[sk] = data
                                single_metadata[mk] = single_subdict
                            else:
                                single_metadata[mk] = mv
                        single_step[k] = single_metadata
                    elif isinstance(v, torch.Tensor):
                        single_step[k] = v[i]
                    else:
                        single_step[k] = v
                self.replay_buffer.add(single_step)

        return step_data

    def collect_trajectory(
        self, max_steps: int, context: Optional[ExecutionContext] = None
    ) -> List[Dict[str, Any]]:
        """Collects a sequence of steps until done or max_steps is reached."""
        trajectory = []
        for _ in range(max_steps):
            step_data = self.step(context)
            trajectory.append(step_data)
            if step_data["done"].any():
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
