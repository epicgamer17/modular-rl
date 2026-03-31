"""
NewSearchPettingZooActor / NewSearchTester:
  Drop-in replacements that wire the NEW search engine into OLD actor/tester classes.

Used by the A/B notebook (Cell 3) to swap OLD search → NEW search while keeping
everything else (PettingZoo env loop, old replay buffer, old executor) unchanged.

Must live in a proper module (not a Jupyter notebook cell) so that
TorchMPExecutor can pickle and spawn worker processes.
"""
from __future__ import annotations

from typing import Any, Callable, Optional
import torch

from old_muzero.agents.workers.actors import PettingZooActor
from old_muzero.agents.workers.tester import Tester
from old_muzero.replay_buffers.modular_buffer import ModularReplayBuffer
from old_muzero.agents.action_selectors.selectors import BaseActionSelector
from old_muzero.modules.agent_nets.modular import ModularAgentNetwork
from old_muzero.agents.action_selectors.policy_sources import SearchPolicySource, NetworkPolicySource


class NewSearchPettingZooActor(PettingZooActor):
    """PettingZooActor that uses the NEW search engine instead of the old one.

    Constructor matches the old actor so TorchMPExecutor._worker_loop can
    instantiate it with the same *args tuple as any other actor.

    The new search is created inside __init__ (i.e. inside the worker process)
    so it never needs to be pickled across process boundaries.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_network: ModularAgentNetwork,
        action_selector: BaseActionSelector,
        replay_buffer: ModularReplayBuffer,
        num_players: Optional[int] = None,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        name: str = "agent",
        *,
        worker_id: int = 0,
        **kwargs,
    ):
        # Pass a NetworkPolicySource so the parent skips old search creation.
        # We replace policy_source with new search below.
        dummy_source = NetworkPolicySource(agent_network)
        super().__init__(
            env_factory,
            agent_network,
            action_selector,
            replay_buffer,
            num_players,
            config,
            device,
            name,
            worker_id=worker_id,
            policy_source=dummy_source,
        )

        # Now replace the old search with the new search engine.
        # Import here (inside worker process) so no pickling is required.
        from agents.factories.search import SearchBackendFactory as NewSearchFactory

        new_search = NewSearchFactory.create(
            config,
            device=self.device,
            num_actions=config.game.num_actions,
        )
        self.policy_source = SearchPolicySource(new_search, self.agent_network, config)


# TorchMPExecutor routes actors by worker_cls.__name__. Make NewSearchPettingZooActor
# appear as "PettingZooActor" so collect_data routing works correctly.
NewSearchPettingZooActor.__name__ = "PettingZooActor"


class NewSearchTester(Tester):
    """Tester that uses the NEW search engine instead of the old one.

    TorchMPExecutor keys signaling and result routing off worker_cls.__name__ == "Tester".
    We set __name__ = "Tester" below so trigger_test / poll_test / request_work all work.

    Constructor signature matches Tester exactly so the old executor can launch it.
    Must live in a module (not a notebook) for multiprocessing pickling.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_network: Any,
        action_selector: BaseActionSelector,
        replay_buffer: Any,
        num_players: int,
        config: Any,
        device: torch.device,
        name: str,
        test_types=None,
        *,
        worker_id: int = 0,
    ):
        # Temporarily disable search so parent uses NetworkPolicySource, then we swap.
        original_search_enabled = getattr(config, "search_enabled", False)
        config.search_enabled = False
        try:
            super().__init__(
                env_factory,
                agent_network,
                action_selector,
                replay_buffer,
                num_players,
                config,
                device,
                name,
                test_types,
                worker_id=worker_id,
            )
        finally:
            config.search_enabled = original_search_enabled

        # Replace policy_source with the new search engine (created here, inside the worker).
        from agents.factories.search import SearchBackendFactory as NewSearchFactory

        new_search = NewSearchFactory.create(
            config,
            device=self.device,
            num_actions=config.game.num_actions,
        )
        self.policy_source = SearchPolicySource(new_search, self.agent_network, config)


# TorchMPExecutor.use_signaling, request_work, and collect_data all key off
# worker_cls.__name__ == "Tester". Masquerade as "Tester" so trigger/signal/result
# routing all work without touching executor code.
NewSearchTester.__name__ = "Tester"
