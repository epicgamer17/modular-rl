from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class TaskType(Enum):
    COLLECT = "collect"
    EVALUATE = "evaluate"
    REANALYZE = "reanalyze"


@dataclass(frozen=True)
class TaskRequest:
    """
    Standardized request object for command-based worker execution.
    """

    task_type: TaskType
    batch_size: int = 0  # Used for steps, episodes, or batch size depending on type
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerPayload:
    """
    Standardized message object for cross-process communication.
    Ensures that all worker outputs are packaged consistently.
    """

    worker_type: str
    metrics: Dict[str, Any]
    data: Optional[Any] = (
        None  # Optional trajectory or model data (e.g. for reanalysis)
    )
