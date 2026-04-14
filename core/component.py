from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Type
from core.contracts import SemanticType

if TYPE_CHECKING:
    from core.blackboard import Blackboard

class PipelineComponent(ABC):
    """
    Base interface for all state and data flow components in the Learner pipeline.
    Components read from and write to the shared 'Blackboard'.
    """
    
    @property
    @abstractmethod
    def requires(self) -> Dict[str, Type[SemanticType]]:
        """Dynamically resolved mapping of keys to semantic types this instance requires."""
        pass

    @property
    @abstractmethod
    def provides(self) -> Dict[str, Type[SemanticType]]:
        """Dynamically resolved mapping of keys to semantic types this instance produces."""
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
