import torch.multiprocessing as mp
import traceback
from typing import Callable, List, Optional
from stats.stats import StatTracker


class TorchMPAgent:
    """
    Mixin class for agents using torch.multiprocessing for parallel environment execution.
    """

    def setup_mp(self):
        """Initializes multiprocessing components."""
        self.stop_flag = mp.Value("i", 0)
        self.error_queue = mp.Queue()
        self.workers: List[mp.Process] = []

    def start_workers(
        self, worker_fn: Callable, num_workers: int, stats_client: StatTracker, **kwargs
    ):
        """
        Starts the worker processes.

        Args:
            worker_fn: The function to run in each worker process.
            num_workers: Number of workers to spawn.
            stats_client: The StatTracker client for the workers.
            **kwargs: Additional arguments for the worker_fn.
        """
        self.workers = [
            mp.Process(
                target=worker_fn,
                args=(i, self.stop_flag, stats_client, self.error_queue),
                kwargs=kwargs,
            )
            for i in range(num_workers)
        ]
        for w in self.workers:
            w.start()

    def stop_workers(self):
        """Stops all running worker processes."""
        self.stop_flag.value = 1
        for w in self.workers:
            if w.is_alive():
                w.terminate()
                w.join()
        self.workers = []

    def check_worker_errors(self):
        """Checks the error queue for any exceptions from worker processes."""
        if not self.error_queue.empty():
            err, tb = self.error_queue.get()

            # Stop all workers immediately
            self.stop_workers()

            print("Error detected in worker process:")
            print("".join(tb))
            raise err

    def __getstate__(self):
        """Excludes multiprocessing objects that cannot be pickled."""
        try:
            state = super().__getstate__()
            state = state.copy()
        except AttributeError:
            state = self.__dict__.copy()

        if "workers" in state:
            del state["workers"]
        if "error_queue" in state:
            del state["error_queue"]
        return state
