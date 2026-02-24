import torch.multiprocessing as mp
import traceback
import time
from typing import Any, Dict, List, Optional, Tuple, Type
from .base import BaseExecutor


class TorchMPExecutor(BaseExecutor):
    """
    Executor that runs workers in separate processes using torch.multiprocessing.
    """

    def __init__(self):
        super().__init__()
        # Set sharing strategy to file_system to avoid torch_shm_manager issues on Mac
        # TODO: MAKE THIS ONLY DO THIS FOR AGENT/SANDBOX RUNS AND NOT WHEN A USER RUNS IT
        try:
            mp.set_sharing_strategy("file_system")
        except Exception:
            pass
        self.stop_flag = mp.Value("i", 0)
        self.result_queue = mp.Queue()
        self.error_queue = mp.Queue()
        self.worker_events = {}  # {worker_type_name: mp.Event}

    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        self.stop_flag.value = 0
        self.workers = []

        # Create a trigger event for this worker type if it doesn't exist
        type_name = worker_cls.__name__
        if type_name not in self.worker_events:
            self.worker_events[type_name] = mp.Event()

        trigger_event = self.worker_events[type_name]

        for i in range(num_workers):
            p = mp.Process(
                target=self._worker_loop,
                args=(
                    worker_cls,
                    args,
                    i,  # Pass worker_id
                    self.stop_flag,
                    self.result_queue,
                    self.error_queue,
                    trigger_event,
                ),
            )
            p.start()
            self.workers.append(p)

    @staticmethod
    def _worker_loop(
        worker_cls,
        args,
        worker_id,
        stop_flag,
        result_queue,
        error_queue,
        trigger_event=None,
    ):
        try:
            worker = worker_cls(*args, worker_id=worker_id)
            worker.setup()

            # Determine if we should use signaling based on the worker type
            # We explicitly want signaling for Tester
            use_signaling = worker_cls.__name__ == "Tester"

            while not stop_flag.value:
                if use_signaling and trigger_event is not None:
                    # Wait for a signal to perform work
                    # Check stop_flag periodically while waiting
                    while not stop_flag.value:
                        if trigger_event.wait(timeout=0.1):
                            trigger_event.clear()
                            break

                    if stop_flag.value:
                        break

                data = worker.play_sequence()
                result_queue.put((worker_cls.__name__, data))
        except Exception as e:
            error_queue.put((e, traceback.format_exc()))
            # We don't re-raise here to avoid polluting stdout,
            # the main process will catch it.
        finally:
            # Ensure resources are cleaned up if worker has a close method
            if "worker" in locals():
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
        # 1. Check if error queue has something
        if not self.error_queue.empty():
            err, tb = self.error_queue.get()
            self.stop()
            print("Error detected in worker process:")
            print("".join(tb))
            raise err

        # 2. Check if all workers are still alive
        for i, w in enumerate(self.workers):
            if not w.is_alive():
                # Check if it exited with code 0 (clean stop) or not
                if (
                    w.exitcode != 0
                    and w.exitcode is not None
                    and self.stop_flag.value == 0
                ):
                    self.stop()
                    raise RuntimeError(
                        f"Worker process {i} died unexpectedly with exit code {w.exitcode}"
                    )

    def update_weights(self, state_dict: Dict[str, Any]):
        # In TorchMP with shared memory, weights are often updated in-place
        # on the shared model. This method is here for compatibility.
        # If the models aren't shared, this would need a different mechanism.
        pass

    def request_work(self, worker_type: Type):
        """Signals the trigger event for the specified worker type."""
        type_name = worker_type.__name__
        if type_name in self.worker_events:
            self.worker_events[type_name].set()

    def stop(self):
        self.stop_flag.value = 1

        # Give workers a moment to see the stop flag
        time.sleep(0.1)

        for w in self.workers:
            if w.is_alive():
                # For some environments, join() might hang if the queue is full
                # terminate() is safer but join() is still needed to clean up
                w.terminate()
                w.join(timeout=1.0)
                if w.is_alive():
                    # Force kill if still alive
                    import os
                    import signal

                    try:
                        os.kill(w.pid, signal.SIGKILL)
                    except:
                        pass
        self.workers = []

        # Close and clear queues to release file descriptors/threads
        for q in [self.result_queue, self.error_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break
            q.close()
            q.join_thread()
