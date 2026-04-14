from typing import Dict, Any, Generator

def infinite_ticks() -> Generator[Dict[str, Any], None, None]:
    """Generates empty dictionaries to trigger the BlackboardEngine indefinitely.

    This function serves as a driver for the BlackboardEngine in actor or 
    learner loops that require continuous execution.

    Yields:
        Dict[str, Any]: A dictionary containing the current "tick" index.
    """
    tick: int = 0
    while True:
        yield {"tick": tick}
        tick += 1
