from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class WorkerPayload:
    """
    Standardized message object for cross-process communication.
    Ensures that all worker outputs are packaged consistently.
    """
    worker_type: str
    metrics: Dict[str, Any]
    data: Optional[Any] = None  # Optional trajectory or model data (e.g. for reanalysis)
