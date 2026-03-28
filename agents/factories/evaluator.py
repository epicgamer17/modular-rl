from typing import Any, List, Optional, Type
from agents.workers.evaluator import BaseTestType, StandardGymTest, SelfPlayTest, VsAgentTest

class TestFactory:
    """Factory for creating test strategies and EvaluatorActor components."""

    @staticmethod
    def create_default_test_types(
        config: Any, num_trials: Optional[int] = None
    ) -> List[BaseTestType]:
        """
        Creates standard test types based on game configuration.
        
        Args:
            config: The global config object.
            num_trials: Optional override for the number of episodes per test.
        """
        test_types = []
        trials = num_trials if num_trials is not None else getattr(config, "test_trials", 5)
        
        # Check if we have multiple players or a specific multi-agent flag
        num_players = getattr(config.game, "num_players", 1)
        
        if num_players > 1:
            # For multi-agent, default to self-play
            test_types.append(SelfPlayTest("self_play", trials))
            
            # If there are specific test_agents (opponents) defined, add VsAgent tests
            test_agents = getattr(config.game, "test_agents", [])
            for agent in test_agents:
                # Add a test for each position the student could take
                for pos in range(num_players):
                    name = f"vs_{agent.name}_p{pos}"
                    test_types.append(VsAgentTest(name, trials, agent, pos))
        else:
            # Single player
            test_types.append(StandardGymTest("standard", trials))
            
        return test_types

    @staticmethod
    def create_custom_test(test_type: str, name: str, num_trials: int, **kwargs) -> BaseTestType:
        """Creates a specific test type by name."""
        if test_type == "standard":
            return StandardGymTest(name, num_trials)
        elif test_type == "self_play":
            return SelfPlayTest(name, num_trials)
        elif test_type == "vs_agent":
            return VsAgentTest(name, num_trials, kwargs["opponent"], kwargs["player_idx"])
        else:
            raise ValueError(f"Unknown test type: {test_type}")
