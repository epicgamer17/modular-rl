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

    def execute(self, blackboard: Blackboard) -> None:
        # Determine if an episode just finished
        done = blackboard.meta.get("done", False)
        
        if done:
            # 1. Standardize sample counting for the Executor
            # The executor uses 'num_samples' to track progress towards min_samples
            ep_len = blackboard.meta.get("episode_length")
            if ep_len is None:
                # Fallback to PettingZoo agent-specific episode step if available
                # Note: blackboard.meta['info']['episode_step'] might exist
                ep_len = blackboard.meta.get("info", {}).get("episode_step", 0)
            
            blackboard.meta["num_samples"] = ep_len
            
            # 2. Consolidated Score (for Multi-agent fallback)
            if "score" not in blackboard.meta:
                # Try to find agent-specific scores injected by components like PettingZooAECComponent
                agent_scores = [v for k, v in blackboard.meta.items() if k.startswith("episode_score_")]
                if agent_scores:
                    # Default to the first agent's score or mean? Usually first agent in registries.
                    blackboard.meta["score"] = agent_scores[0]
                elif "episode_score" in blackboard.meta:
                    blackboard.meta["score"] = blackboard.meta["episode_score"]

            # 3. Throughput / FPS
            t_now = time.perf_counter()
            duration = t_now - self._last_time
            self._last_time = t_now
            
            if duration > 0 and blackboard.meta["num_samples"] > 0:
                blackboard.meta["fps"] = blackboard.meta["num_samples"] / duration

            # 4. Search stats (if available in meta)
            # These are often injected by SearchInferenceComponent
            sm = blackboard.meta.get("search_metadata")
            if sm and isinstance(sm, dict):
                # SearchInferenceComponent might have already put things in search_metadata
                # We can promote them to top-level for easier logging if desired.
                pass
