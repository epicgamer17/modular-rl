import time
import torch
from typing import Dict, Any, Optional, Set
from core import PipelineComponent, Blackboard
from core.contracts import Key, Reward, Done, LossScalar, SemanticType, Metric


class TelemetryComponent(PipelineComponent):
    """
    Component responsible for calculating and storing telemetry/stats in blackboard.meta.
    Moves knowledge of 'episode_length', 'score', etc. out of the Executor and into the Pipeline.

    Supports Universal Time Mandate: Handles 'reward' and 'done' as tensors of shape [B, T, ...].
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._last_time = time.perf_counter()
        self.episode_reward: Optional[torch.Tensor] = None
        self.episode_length: Optional[torch.Tensor] = None

        # Deterministic contracts computed at initialization
        self._requires = {Key("data.reward", Reward), Key("data.done", Done)}
        self._provides = {
            Key("meta.score", Metric): "optional",
            Key("meta.episode_length", Metric): "optional",
            Key("meta.fps", Metric): "optional",
            Key("meta.running_reward", Metric): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures reward and done signals are accessible."""
        assert blackboard.meta.get("reward") is not None or blackboard.data.get("reward") is not None, (
            "TelemetryComponent: 'reward' not found in blackboard.meta or blackboard.data"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        # 1. Extraction and Normalization
        reward = blackboard.meta.get("reward", 0.0)
        done = blackboard.meta.get("done", False)

        if not torch.is_tensor(reward):
            reward = torch.as_tensor(reward, dtype=torch.float32)
        if not torch.is_tensor(done):
            done = torch.as_tensor(done, dtype=torch.bool)

        # Force [B, T] shape for calculations
        # [] -> [1, 1], [B] -> [B, 1]
        if reward.ndim == 0:
            reward = reward.view(1, 1)
        elif reward.ndim == 1:
            reward = reward.view(-1, 1)

        if done.ndim == 0:
            done = done.view(1, 1)
        elif done.ndim == 1:
            done = done.view(-1, 1)

        B, T = reward.shape

        # 2. State Initialization (Lazy)
        if self.episode_reward is None or self.episode_reward.shape[0] != B:
            self.episode_reward = torch.zeros(B, device=reward.device, dtype=torch.float32)
            self.episode_length = torch.zeros(B, device=reward.device, dtype=torch.long)

        # 3. Update Running Stats
        # We sum over the time dimension T if a sequence is provided
        self.episode_reward += reward.sum(dim=1)
        self.episode_length += T

        # Env-level finishes: shape [B]
        any_done = done.any(dim=1)
        outputs = {}

        if any_done.any():
            indices = torch.where(any_done)[0]
            
            # Extract final stats for exactly those environments that finished
            finished_scores = self.episode_reward[indices]
            finished_lengths = self.episode_length[indices]

            # Aggregate scores for logging (mean of episodes finished in this tick)
            mean_score = finished_scores.mean().item()
            mean_length = finished_lengths.float().mean().item()

            outputs.update({
                "meta.num_samples": mean_length,
                "meta.score": mean_score,
                "meta.episode_score": mean_score,
                "meta.episode_length": mean_length
            })

            # Throughput / FPS calculation
            t_now = time.perf_counter()
            duration = t_now - self._last_time
            self._last_time = t_now

            if duration > 0:
                # Total steps across all finished environments / time taken
                outputs["meta.fps"] = finished_lengths.sum().item() / duration

            # 5. Reset tracking only for those specific environments
            self.episode_reward[indices] = 0.0
            self.episode_length[indices] = 0

        # Update running stats for output after potential resets
        outputs.update({
            "meta.running_reward": self.episode_reward.mean().item(),
            "meta.running_length": self.episode_length.float().mean().item()
        })

        # Promote search metadata if available
        sm = blackboard.meta.get("search_metadata")
        if sm and isinstance(sm, dict):
            outputs.update({f"meta.search.{k}": v for k, v in sm.items()})

        return outputs


class SequenceTerminatorComponent(PipelineComponent):
    """
    Signals the BlackboardEngine to stop execution for the current sequence
    when a 'done' signal is detected in the blackboard.
    
    This is essential for play_sequence() loops in executors to return 
    the final telemetry of an episode and proceed to weight updates.

    Supports Universal Time Mandate: Triggers if ANY environment in a batch is done.
    """
    def __init__(self):
        self._requires = {Key("data.done", Done)}
        self._provides = {Key("meta.stop_execution", Metric): "optional"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """No strict validation needed; component gracefully handles missing done signal."""
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        done = blackboard.meta.get("done")
        if done is None:
            done = blackboard.data.get("done")

        if done is not None:
            if torch.is_tensor(done):
                if done.any():
                    return {"meta.stop_execution": True}
            elif done:
                return {"meta.stop_execution": True}
        
        return {}
