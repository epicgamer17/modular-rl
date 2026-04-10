import time
from typing import Dict, Any, Optional
from core import PipelineComponent, Blackboard


class TelemetryComponent(PipelineComponent):
    """
    Component responsible for calculating and storing telemetry/stats in blackboard.meta.
    Moves knowledge of 'episode_length', 'score', etc. out of the Executor and into the Pipeline.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._last_time = time.perf_counter()
        self.episode_reward = 0.0
        self.episode_length = 0

    def execute(self, blackboard: Blackboard) -> None:
        # Track running stats
        reward = blackboard.meta.get("reward", 0.0)
        self.episode_reward += float(reward)
        self.episode_length += 1

        # Determine if an episode just finished
        done = blackboard.meta.get("done", False)

        # Expose running stats to Blackboard (internal metrics)
        blackboard.meta["running_reward"] = self.episode_reward
        blackboard.meta["running_length"] = self.episode_length

        if done:
            # Finalize telemetry for the finished episode
            blackboard.meta["num_samples"] = self.episode_length
            blackboard.meta["score"] = self.episode_reward

            # Regression test compatibility: only set this when done
            blackboard.meta["episode_score"] = self.episode_reward
            blackboard.meta["episode_length"] = self.episode_length

            # Throughput / FPS calculation
            t_now = time.perf_counter()
            duration = t_now - self._last_time
            self._last_time = t_now

            if duration > 0 and self.episode_length > 0:
                blackboard.meta["fps"] = self.episode_length / duration

            # Reset tracking for next episode
            self.episode_reward = 0.0
            self.episode_length = 0

        # Promote search metadata if available
        sm = blackboard.meta.get("search_metadata")
        if sm and isinstance(sm, dict):
            pass
