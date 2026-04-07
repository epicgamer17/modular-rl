from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learner.core import Blackboard

class PipelineComponent(ABC):
    """
    Base interface for all state and data flow components in the Learner pipeline.
    Components read from and write to the shared 'Blackboard'.
    """
    
    @abstractmethod
    def execute(self, blackboard: 'Blackboard') -> None:
        """
        Execute this component's logic, mutating the blackboard in place.
        """
        pass
