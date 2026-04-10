from typing import Any, Dict, Iterable, Optional
import torch

from core import BlackboardEngine, infinite_ticks

class ActorWorker:
    """
    A lightweight hardware wrapper for the BlackboardEngine.
    The Executor interacts with this worker, which simply orchestrates
    the engine's execution loop.
    """

    def __init__(
        self,
        engine: BlackboardEngine,
        env_iterator: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        self.engine = engine
        self.env_iterator = env_iterator if env_iterator is not None else infinite_ticks()
        self.worker_id = kwargs.get("worker_id", 0)
        # Expose engine attributes for easier access by executors if needed
        self.device = engine.device

    def setup(self) -> None:
        """Standard hook for executors. Setup is typically handled by components."""
        pass

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Updates parameters for the components.
        Note: Neural network weights are typically shared via memory references.
        """
        # If any component needs explicit parameter updates (not via reference),
        # we can iterate through self.engine.components here.
        pass

    def play_sequence(self) -> Dict[str, Any]:
        """
        Runs the engine until an episode or task signals completion.

        Returns:
            The meta dictionary from the terminal blackboard state, 
            containing episode statistics and telemetry.
        """
        for result in self.engine.step(self.env_iterator):
            if result["meta"].get("stop_execution"):
                # Clean up the flag for the next call if this worker is reused
                result["meta"].pop("stop_execution", None)
                return result["meta"]
        
        return {}
