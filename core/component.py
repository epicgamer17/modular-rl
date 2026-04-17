from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Set, Optional, Any, Dict
from core.contracts import Key, WriteMode

if TYPE_CHECKING:
    from core.blackboard import Blackboard

class PipelineComponent(ABC):
    """
    Base interface for all state and data flow components in the Learner pipeline.
    Components read from and write to the shared 'Blackboard'.
    """

    required: bool = False


    
    @property
    @abstractmethod
    def requires(self) -> Set[Key]:
        """
        Required keys for this component. 
        MUST be deterministic after initialization. 
        Recommended: Compute once in __init__ and return a private attribute.
        """
        pass

    @property
    @abstractmethod
    def provides(self) -> Dict[Key, WriteMode]:
        """
        Keys produced by this component and their write modes. 
        MUST be deterministic after initialization. 
        Recommended: Compute once in __init__ and return a private attribute.
        
        Write Modes:
        - NEW: Key must not already exist (default).
        - OVERWRITE: Key must exist.
        - APPEND: Data is added to an existing collection.
        - OPTIONAL: Key may or may not be produced.
        """
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
    def execute(self, blackboard: "Blackboard") -> Dict[str, Any]:
        """
        Execute this component's logic and return a dictionary of changes
        to be written by the framework. 
        
        STRICT RULES:
        1. In-place mutation of the 'blackboard' argument is FORBIDDEN.
           The framework passes a frozen view; attempting to mutate will raise a TypeError.
        2. All outputs MUST be returned as a dictionary of path-based updates.
        
        Example Return: {"losses.value_loss": tensor}
        """
        pass
