from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.blackboard import Blackboard

class PipelineComponent(ABC):
    """
    Base interface for all state and data flow components in the Learner pipeline.
    Components read from and write to the shared 'Blackboard'.
    """
    
    @property
    @abstractmethod
    def reads(self) -> set[str]:
        """Dynamically resolved set of keys this instance requires."""
        pass

    @property
    @abstractmethod
    def writes(self) -> set[str]:
        """Dynamically resolved set of keys this instance produces."""
        pass

    @abstractmethod
    def execute(self, blackboard: "Blackboard") -> None:
        """
        Execute this component's logic, mutating the blackboard in place.
        """
        pass
