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
        """
        Args:
            replay_buffer: If set, each step's batched ``step_data`` is unbatched
                and written transition-by-transition via ``replay_buffer.add``.
            recording_fn: Observational hook invoked with each unbatched transition
                on every step. Runs in addition to (not instead of) the replay
                buffer write, so users can log/print without reimplementing the
                unbatching logic.
        """
        from runtime.io.environment import wrap_env

        self.interact_graph = interact_graph
        self.env = wrap_env(env)
        self.recording_fn = recording_fn
        self.replay_buffer = replay_buffer
        self.current_obs = None
        self.last_obs = None
        self.num_envs = self.env.num_envs
        self.last_done = torch.zeros(self.num_envs, dtype=torch.bool)
        self.last_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
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
        from runtime.refs import Value

        # TODO: this shouldnt really be necessary, and the fact that it is, is concerning. See if these kinds of contracts can be improved
        action_data = action_res.data if isinstance(action_res, Value) else action_res

        # TODO: this is also not great, the actor node should really be returning the action directly. or at least these isinstance checks should be unecessary
        if isinstance(action_data, dict):
            action = action_data["action"]
        else:
            action = action_data

        step_res = self.env.step(action)

        from runtime.io.environment import validate_step_result

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

        # Handle truncation bootstrapping: if truncated, next_obs should be the final_observation
        # from the info dict (standard gymnasium behavior for auto-resetting vector envs)
        real_next_obs = step_res.obs.clone()
        for i in range(self.num_envs):
            if step_res.truncated[i] and "final_observation" in step_res.info[i]:
                final_obs = step_res.info[i]["final_observation"]
                if not isinstance(final_obs, torch.Tensor):
                    final_obs = torch.from_numpy(final_obs).to(torch.float32)
                real_next_obs[i] = final_obs

        from core.batch import TransitionBatch
        step_data = TransitionBatch(
            obs=self.current_obs,
            action=action,
            reward=step_res.reward,
            next_obs=real_next_obs,
            done=done.float(),
            terminated=step_res.terminated.float(),
            truncated=step_res.truncated.float(),
            metadata={
                "step_index": context.actor_step,
                "episode_id": context.episode_count,
                "episode_step": context.episode_step,
                "actor_results": results,
                "context": context.to_dict(),
            },
        )

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

        self.last_obs = real_next_obs
        self.last_done = done
        self.last_terminated = step_res.terminated

        # 5. Unbatch and handle recording/buffer write
        if self.replay_buffer is not None or self.recording_fn:
            batch_size = step_data.obs.shape[0]
            for i in range(batch_size):
                single_step = self._unbatch_step_data(step_data, i)
                if self.replay_buffer is not None:
                    self.replay_buffer.add(single_step)
                if self.recording_fn:
                    self.recording_fn(single_step)

        return step_data

    def _unbatch_step_data(self, step_data: "TransitionBatch", index: int) -> "TransitionBatch":
        """Extract a single transition from batched step_data."""
        from core.batch import TransitionBatch
        from runtime.refs import Value

        batch_size = step_data.obs.shape[0]
        
        def unbatch_val(v):
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == batch_size:
                return v[index]
            return v

        # Unbatch metadata
        single_metadata: Dict[str, Any] = {"env_idx": index}
        if step_data.metadata:
            for mk, mv in step_data.metadata.items():
                m_data = mv.data if hasattr(mv, "has_data") and mv.has_data else mv
                if isinstance(m_data, torch.Tensor) and m_data.ndim > 0 and m_data.shape[0] == batch_size:
                    single_metadata[mk] = m_data[index]
                elif isinstance(m_data, dict):
                    single_subdict = {}
                    for sk, sv in m_data.items():
                        s_data = sv.data if hasattr(sv, "has_data") and sv.has_data else sv
                        if isinstance(s_data, torch.Tensor) and s_data.ndim > 0 and s_data.shape[0] == batch_size:
                            single_subdict[sk] = s_data[index]
                        else:
                            single_subdict[sk] = s_data
                    single_metadata[mk] = single_subdict
                else:
                    single_metadata[mk] = m_data

        return TransitionBatch(
            obs=unbatch_val(step_data.obs),
            action=unbatch_val(step_data.action),
            reward=unbatch_val(step_data.reward),
            next_obs=unbatch_val(step_data.next_obs),
            done=unbatch_val(step_data.done),
            terminated=unbatch_val(step_data.terminated),
            truncated=unbatch_val(step_data.truncated),
            log_prob=unbatch_val(step_data.log_prob),
            value=unbatch_val(step_data.value),
            policy_version=unbatch_val(step_data.policy_version),
            metadata=single_metadata
        )

    def collect_trajectory(
        self, max_steps: int, context: Optional[ExecutionContext] = None
    ) -> List[Dict[str, Any]]:
        """Collects a sequence of steps until done or max_steps is reached."""
        trajectory = []
        for _ in range(max_steps):
            step_data = self.step(context)
            trajectory.append(step_data)
            if step_data.done.any():
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
