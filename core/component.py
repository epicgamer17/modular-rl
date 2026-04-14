from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Set
from core.contracts import Key

if TYPE_CHECKING:
    from core.blackboard import Blackboard

class PipelineComponent(ABC):
    """
    Base interface for all state and data flow components in the Learner pipeline.
    Components read from and write to the shared 'Blackboard'.
    """
    
    @property
    @abstractmethod
    def requires(self) -> Set[Key]:
        """Dynamically resolved set of keys this instance requires."""
        pass

    @property
    @abstractmethod
    def provides(self) -> Set[Key]:
        """Dynamically resolved set of keys this instance produces."""
        pass

    @property
    def constraints(self) -> list[str]:
        """Declarative constraints describing the component's data relationships."""
        return []

    def validate(self, blackboard: "Blackboard") -> None:
        """
        Programmatic validation of inputs/outputs on the blackboard.
        Source of truth for correctness.
        """
        pass

    @abstractmethod
    def execute(self, blackboard: "Blackboard") -> None:
        """
        Execute this component's logic, mutating the blackboard in place.
        """
        pass
