import torch.multiprocessing as mp
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type
from .base import BaseExecutor


class TorchMPExecutor(BaseExecutor):
    """
    Executor that runs workers in separate processes using torch.multiprocessing.
    """

    def __init__(self):
        super().__init__()
        self.stop_flag = mp.Value("i", 0)
        self.result_queue = mp.Queue()
        self.error_queue = mp.Queue()

    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        self.stop_flag.value = 0
        self.workers = []
        for i in range(num_workers):
            p = mp.Process(
                target=self._worker_loop,
                args=(
                    worker_cls,
                    args,
                    self.stop_flag,
                    self.result_queue,
                    self.error_queue,
                ),
            )
            p.start()
            self.workers.append(p)

    @staticmethod
    def _worker_loop(worker_cls, args, stop_flag, result_queue, error_queue):
        try:
            worker = worker_cls(*args)
            if hasattr(worker, "setup"):
                worker.setup()

            while not stop_flag.value:
                if hasattr(worker, "play_game"):
                    data = worker.play_game()
                    result_queue.put(data)
                else:
                    raise AttributeError(
                        f"Worker {worker_cls} must implement play_game()"
                    )
        except Exception as e:
            error_queue.put((e, traceback.format_exc()))
            # We don't re-raise here to avoid polluting stdout,
            # the main process will catch it.
        finally:
            # Ensure resources are cleaned up if worker has a close method
            if (
                "worker" in locals()
                and hasattr(worker, "env")
                and hasattr(worker.env, "close")
            ):
                worker.env.close()

    def _fetch_available_results(self) -> List[Any]:
        self._check_errors()
        results = []
        while not self.result_queue.empty():
            try:
                # Use non-blocking get
                results.append(self.result_queue.get_nowait())
            except mp.queues.Empty:
                break
        return results

    def _check_errors(self):
        if not self.error_queue.empty():
            err, tb = self.error_queue.get()
            self.stop()
            print("Error detected in worker process:")
            print("".join(tb))
            raise err

    def update_weights(self, state_dict: Dict[str, Any]):
        # In TorchMP with shared memory, weights are often updated in-place
        # on the shared model. This method is here for compatibility.
        # If the models aren't shared, this would need a different mechanism.
        pass

    def stop(self):
        self.stop_flag.value = 1
        for w in self.workers:
            if w.is_alive():
                w.terminate()
                w.join()
        self.workers = []

        # Clear queues
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                break
        while not self.error_queue.empty():
            try:
                self.error_queue.get_nowait()
            except:
                break
