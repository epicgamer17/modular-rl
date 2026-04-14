from typing import TYPE_CHECKING, Any, Set, Dict
from core.component import PipelineComponent
from core.contracts import (
    Key,
    Observation,
    ActionDistribution,
    ValueEstimate,
    PolicyLogits,
    Metric,
)

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

        # Deterministic contracts computed at initialization
        self._requires = {Key("data.obs", Observation)}
        self._provides = {
            Key("predictions.search_policy", PolicyLogits): "new",
            Key("predictions.search_target_policy", PolicyLogits): "new",
            Key("predictions.search_value", ValueEstimate): "new",
            Key("meta.mcts_simulations", Metric): "new",
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: "Blackboard") -> None:
        """Ensures observations exist and have at least 2 dims."""
        from core.validation import assert_in_blackboard, assert_is_tensor

        assert_in_blackboard(blackboard, "data.obs")
        obs = blackboard.data["obs"]
        assert_is_tensor(obs, msg="for MCTSSearchComponent")
        assert (
            obs.ndim >= 2
        ), f"Observation must have at least 2 dims (B, *), got {obs.ndim}"

    def execute(self, blackboard: "Blackboard") -> Dict[str, Any]:
        """
        Execute search and return the updates for the blackboard.

        Returns:
            Dictionary of blackboard mutations.
        """
        obs = blackboard.data["obs"]
        info = blackboard.data.get("info", {})

        # Determine player_id/to_play if not in info
        if "player" not in info:
            player_id = blackboard.data.get("player_id", 0)
            info = {**info, "player": player_id}

        # Check for tournament end / terminal state
        if (
            blackboard.data.get("done")
            or blackboard.data.get("terminated")
            or blackboard.meta.get("done")
        ):
            return {"meta.stop_execution": True}

        # --- STABILITY GUARD: ENSURE EVAL MODE ---
        # When using BatchNorm, we MUST be in eval mode during search to prevent
        # small MCTS batches from polluting the running statistics and slowing down learning.
        was_training = self.agent_network.training
        self.agent_network.eval()

        # Note: ModularSearch.run returns (root_value, exploratory_policy, target_policy, best_action, search_metadata)
        results = self.search_engine.run(obs, info, self.agent_network)

        if was_training:
            self.agent_network.train()

        updates = {}
        if len(results) == 5:
            root_value, exploratory_policy, target_policy, best_action, search_meta = (
                results
            )

            # --- ABSOLUTE SEARCH CONTRACT ---
            updates["predictions.search_policy"] = exploratory_policy
            updates["predictions.search_target_policy"] = target_policy
            updates["predictions.search_value"] = root_value
        elif len(results) == 3:
            policy, value, search_meta = results
            # Fallback if the search engine doesn't split them
            updates["predictions.search_policy"] = policy
            updates["predictions.search_target_policy"] = policy
            updates["predictions.search_value"] = value
        else:
            raise ValueError(
                f"Unexpected number of return values from search_engine.run: {len(results)}"
            )

        if isinstance(search_meta, dict) and "simulations" in search_meta:
            updates["meta.mcts_simulations"] = search_meta["simulations"]
        elif isinstance(search_meta, dict) and "num_simulations" in search_meta:
            updates["meta.mcts_simulations"] = search_meta["num_simulations"]
        else:
            updates["meta.mcts_simulations"] = getattr(
                self.search_engine, "num_simulations", None
            )

        return updates
