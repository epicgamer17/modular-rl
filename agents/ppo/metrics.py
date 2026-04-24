from typing import Dict, Any

class PPOMetrics:
    """Helper class for PPO metrics."""
    def __init__(self):
        self.stats = {}

    def update(self, results: Dict[str, Any]):
        """Update metrics with results from a training step."""
        if "ppo" in results and results["ppo"].has_data:
            self.stats["loss"] = results["ppo"].data
            
        if "opt" in results and results["opt"].has_data:
            self.stats["last_opt_step"] = results["opt"].data
            
    def get_report(self) -> Dict[str, Any]:
        """Return a report of current metrics."""
        return self.stats
