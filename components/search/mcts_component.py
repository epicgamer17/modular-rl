import torch
from typing import TYPE_CHECKING, Any
from core.component import PipelineComponent

if TYPE_CHECKING:
    from core.blackboard import Blackboard

class MCTSSearchComponent(PipelineComponent):
    """
    Component that executes MCTS search on the current observation.
    
    In ECS, MCTS is just a component that intercepts the Blackboard.
    It reads 'obs' and 'info' from blackboard.data, runs search via the search_engine,
    and writes 'policy' and 'value' directly to blackboard.predictions.
    """
    def __init__(self, search_engine: Any, agent_network: Any):
        """
        Initialize the MCTS search component.
        
        Args:
            search_engine: The search engine instance (e.g., ModularSearch).
            agent_network: The agent network to use for search evaluations.
        """
        self.search_engine = search_engine
        self.agent_network = agent_network

    def execute(self, blackboard: "Blackboard") -> None:
        """
        Execute search and update the blackboard with results.
        """
        # Fails fast if previous components didn't provide obs
        if "obs" not in blackboard.data:
            raise KeyError("Blackboard data missing 'obs'. Ensure an observation provider ran before MCTSSearchComponent.")
            
        obs = blackboard.data["obs"] 
        info = blackboard.data.get("info", {})
        
        # Determine player_id/to_play if not in info
        if "player" not in info:
            player_id = blackboard.data.get("player_id", 0)
            info = {**info, "player": player_id}

        # Check for tournament end / terminal state
        if blackboard.data.get("done") or blackboard.data.get("terminated") or blackboard.meta.get("done"):
            blackboard.meta["stop_execution"] = True
            return
        # Note: ModularSearch.run returns (root_value, exploratory_policy, target_policy, best_action, search_metadata)
        results = self.search_engine.run(obs, info, self.agent_network)
        
        if len(results) == 5:
            root_value, exploratory_policy, target_policy, best_action, search_meta = results
            policy = target_policy
            value = root_value
        elif len(results) == 3:
            policy, value, search_meta = results
        else:
            raise ValueError(f"Unexpected number of return values from search_engine.run: {len(results)}")
            
        blackboard.predictions["policy"] = policy
        blackboard.predictions["probs"] = policy  # For ActionSelector compatibility
        blackboard.predictions["value"] = value
        
        # Also write to logits if preferred by selector (MCTS uses probs, but we can set logits to probs for convenience)
        blackboard.predictions["logits"] = torch.log(policy + 1e-8)
        if isinstance(search_meta, dict) and "simulations" in search_meta:
            blackboard.meta["mcts_simulations"] = search_meta["simulations"]
        elif isinstance(search_meta, dict) and "num_simulations" in search_meta:
            blackboard.meta["mcts_simulations"] = search_meta["num_simulations"]
        else:
            blackboard.meta["mcts_simulations"] = getattr(self.search_engine, "num_simulations", None)
