"""Model parameter state."""

from typing import Dict
import torch


class ParameterStore:
    """
    Manages model state (parameters and buffers) and versioning.
    """

    def __init__(self, state: Dict[str, torch.Tensor]):
        self._state = state
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Returns the current state (parameters and buffers)."""
        return self._state

    def update_state(self, new_state: Dict[str, torch.Tensor]) -> None:
        """
        Updates state in-place and increments version.
        Explicitly uses .copy_() to maintain reference integrity if needed.
        """
        with torch.no_grad():
            for name, tensor in new_state.items():
                if name in self._state:
                    self._state[name].copy_(tensor)
                else:
                    self._state[name] = tensor.detach().clone()
        self._version += 1
