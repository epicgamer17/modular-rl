import time
import torch
from typing import Dict, Any, Optional, Set
from core import PipelineComponent, Blackboard
from core.contracts import Key, Reward, Done, LossScalar, SemanticType


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

    @property
    def requires(self) -> Set[Key]:
        return {Key("data.reward", Reward), Key("data.done", Done)}

    @property
    def provides(self) -> Set[Key]:
        return {
            Key("meta.score", LossScalar),
            Key("meta.episode_length", LossScalar),
            Key("meta.fps", LossScalar),
            Key("meta.running_reward", LossScalar),
        }

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
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

        # 4. Finalize Statistics for Finished Episodes
        any_done = done.any(dim=1)  # [B]

        if any_done.any():
            indices = torch.where(any_done)[0]
            
            # Extract final stats for exactly those environments that finished
            finished_scores = self.episode_reward[indices]
            finished_lengths = self.episode_length[indices]

            # Aggregate scores for logging (mean of episodes finished in this tick)
            mean_score = finished_scores.mean().item()
            mean_length = finished_lengths.float().mean().item()

            blackboard.meta["num_samples"] = mean_length
            blackboard.meta["score"] = mean_score
            blackboard.meta["episode_score"] = mean_score
            blackboard.meta["episode_length"] = mean_length

            # Throughput / FPS calculation
            t_now = time.perf_counter()
            duration = t_now - self._last_time
            self._last_time = t_now

            if duration > 0:
                # Total steps across all finished environments / time taken
                blackboard.meta["fps"] = finished_lengths.sum().item() / duration

            # 5. Reset tracking only for those specific environments
            self.episode_reward[indices] = 0.0
            self.episode_length[indices] = 0

        # Expose current running stats (mean across all active environments)
        # We do this after reset so that if an episode just finished, it starts from 0 for the next step's telemetry
        blackboard.meta["running_reward"] = self.episode_reward.mean().item()
        blackboard.meta["running_length"] = self.episode_length.float().mean().item()

        # Promote search metadata if available
        sm = blackboard.meta.get("search_metadata")
        if sm and isinstance(sm, dict):
            pass


class SequenceTerminatorComponent(PipelineComponent):
    """
    Signals the BlackboardEngine to stop execution for the current sequence
    when a 'done' signal is detected in the blackboard.
    
    This is essential for play_sequence() loops in executors to return 
    the final telemetry of an episode and proceed to weight updates.

    Supports Universal Time Mandate: Triggers if ANY environment in a batch is done.
    """
    @property
    def requires(self) -> Set[Key]:
        return {Key("data.done", Done)}

    @property
    def provides(self) -> Set[Key]:
        return {Key("meta.stop_execution", SemanticType)}

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        done = blackboard.meta.get("done")
        if done is None:
            done = blackboard.data.get("done")

        if done is not None:
            if torch.is_tensor(done):
                if done.any():
                    blackboard.meta["stop_execution"] = True
            elif done:
                blackboard.meta["stop_execution"] = True
