from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from typing import Optional

class ProgressTracker:
    """Handles beautiful progress bars for long-running tasks."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        self._tasks = {}

    def add_task(self, description: str, total: int) -> int:
        task_id = self.progress.add_task(description, total=total)
        self._tasks[description] = task_id
        return task_id

    def update(self, task_id: int, advance: int = 1, **kwargs):
        self.progress.update(task_id, advance=advance, **kwargs)

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()

def create_progress_bar() -> ProgressTracker:
    return ProgressTracker()
