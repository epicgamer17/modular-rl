import time
from typing import Any, Dict, List, Tuple
from agents.workers.puffer_actor import PufferActor
from agents.executors.base import BaseExecutor


class PufferExecutor(BaseExecutor):
    """
    Executor for PufferLib based environments.
    Spawns multiple PufferActor processes which write directly to the shared buffer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _launch_workers(self, worker_cls, args, num_workers):
        """
        Launches PufferActor processes.
        """
        # args contains (env_factory, network, selector, buffer, num_players, config, ...)
        env_factory, network, selector, buffer = args[0], args[1], args[2], args[3]
        # PufferActor is specialized and might expect these explicitly or we pass them correctly
        for _ in range(num_workers):
            actor = PufferActor(
                config=self.config,
                env_creator=env_factory,
                shared_network=network,
                shared_buffer=buffer,
                search_alg=selector,  # Assuming selector acts as search_alg for Puffer
            )
            actor.start()
            self.workers.append(actor)

    def _fetch_available_results(self) -> List[Any]:
        """PufferActors write directly to the buffer, so no results are fetched via queue."""
        time.sleep(0.01)
        return []

    def stop(self):
        """Stops all PufferActor processes."""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
                if worker.is_alive():
                    import os
                    import signal

                    try:
                        os.kill(worker.pid, signal.SIGKILL)
                    except Exception:
                        pass
        self.workers = []

    def update_weights(self, state_dict, params=None):
        """No-op. Assuming shared_network handles updates automatically from learner."""
        pass

    def request_work(self, worker_type):
        """No-op for PufferActor."""
        pass
