import queue
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
        self.param_queue = mp.Queue()
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
                    self.param_queue,
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
        param_queue,
        trigger_event=None,
    ):
        # Configure thread affinity to avoid OpenMP contention
        import os
        import torch

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        # Stagger start times to avoid overwhelming the compiler (and avoid race conditions in Triton cache)
        config = args[5]
        if config.compilation.enabled:
            time.sleep(worker_id * 1.0)
        elif worker_id > 0:
            time.sleep(worker_id * 0.1)

        try:
            worker = worker_cls(*args, worker_id=worker_id)
            worker.setup()

            # Determine if we should use signaling based on the worker type
            # We explicitly want signaling for Tester
            use_signaling = worker_cls.__name__ == "Tester"

            while not stop_flag.value:
                # Check for parameter updates
                while not param_queue.empty():
                    try:
                        params = param_queue.get_nowait()
                        if params is not None:
                            worker.update_parameters(params)
                            if params.get("reset_noise") and hasattr(
                                worker.agent_network, "reset_noise"
                            ):
                                worker.agent_network.reset_noise()
                            if hasattr(worker, "action_selector"):
                                worker.action_selector.update_parameters(params)
                    except queue.Empty:
                        break

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
            # We stringify the exception to prevent obscure pickling errors (like 'cell' objects)
            # from masking the real underlying crash when this goes through the mp.Queue.
            error_queue.put((f"{type(e).__name__}: {str(e)}", traceback.format_exc()))
            # We don't re-raise here to avoid polluting stdout,
            # the main process will catch it.
        finally:
            # Ensure resources are cleaned up if worker has a close method
            if "worker" in locals():
                if hasattr(worker, "env") and worker.env is not None:
                    try:
                        worker.env.close()
                    except:
                        pass
                if hasattr(worker, "vec_env") and worker.vec_env is not None:
                    try:
                        worker.vec_env.close()
                    except:
                        pass

    def _fetch_available_results(self) -> List[Any]:
        self._check_errors()
        results = []
        while not self.result_queue.empty():
            try:
                # Use non-blocking get
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _check_errors(self):
        # 1. Check if error queue has something
        if not self.error_queue.empty():
            err, tb = self.error_queue.get()
            self.stop()
            print("Error detected in worker process:")
            print("".join(tb))
            # err is a stringified exception message (intentionally serialized as a string
            # in _worker_loop to avoid pickling issues with exotic exception types).
            # Wrap it in RuntimeError so it is a proper BaseException subclass.
            if not isinstance(err, BaseException):
                err = RuntimeError(err)
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

    def update_weights(
        self, state_dict: Dict[str, Any], params: Optional[Dict[str, Any]] = None
    ):
        # In TorchMP with shared memory, weights are updated in-place on the shared model.
        # But wait, we must update the uncompiled underlying model if it's compiled, relying on Pytorch references.
        # Actually, if they share memory, we don't need to load_state_dict here!
        # The main thread does step() on its model, which IS the shared model, since both are uncompiled originally.
        # However, to be safe, we just signal parameter updates.
        if params is None:
            params = {}

        # Signal noise reset if this is a learning update
        params["reset_noise"] = True

        # Send to all workers via the queue
        for _ in range(len(self.workers)):
            self.param_queue.put(params)

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
