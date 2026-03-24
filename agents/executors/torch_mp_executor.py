import queue
import torch.multiprocessing as mp
import traceback
import time
from typing import Any, Dict, List, Optional, Tuple, Type
from .base import BaseExecutor
from agents.workers.payloads import WorkerPayload
import torch


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
        self.command_queue = mp.Queue()
        self.worker_events = {}  # {worker_type_name: mp.Event}
        self.shared_networks = set()  # To avoid redundant updates to shared RAM

    def _launch_workers(self, worker_cls: Type, args: Tuple, num_workers: int):
        self.stop_flag.value = 0
        if not hasattr(self, "workers"):
            self.workers = []

        # Create a trigger event for this worker type if it doesn't exist
        type_name = worker_cls.__name__
        if type_name not in self.worker_events:
            self.worker_events[type_name] = mp.Event()

        trigger_event = self.worker_events[type_name]

        # Track network if it is likely shared (index 2 of standard worker args)
        if len(args) > 2 and isinstance(args[2], torch.nn.Module):
            self.shared_networks.add(args[2])

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
                    self.command_queue,
                    trigger_event,
                ),
            )
            p.start()
            self.workers.append((worker_cls, p))

    @staticmethod
    def _worker_loop(
        worker_cls,
        args,
        worker_id,
        stop_flag,
        result_queue,
        error_queue,
        param_queue,
        command_queue,
        trigger_event=None,
    ):
        # Configure thread affinity to avoid OpenMP contention
        import os
        import torch

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        # Stagger start times to avoid overwhelming the compiler (and avoid race conditions in Triton cache)
        # We try to find the config object in args (usually at index 3, 4 or 5)
        config = None
        for idx in [3, 4, 5]:
            if len(args) > idx:
                potential_config = args[idx]
                if hasattr(potential_config, "compilation"):
                    config = potential_config
                    break

        if config and config.compilation.enabled:
            time.sleep(worker_id * 1.0)
        elif worker_id > 0:
            time.sleep(worker_id * 0.1)

        try:
            worker = worker_cls(*args, worker_id=worker_id)
            worker.setup()

            # Signaling: Only wait if worker has synchronous methods (collect/evaluate/reanalyze)
            # Most modern actors are now synchronous by default in the training loop.
            use_signaling = True

            while not stop_flag.value:
                # 1. Parameter Updates
                while not param_queue.empty():
                    try:
                        update_dict = param_queue.get_nowait()
                        if update_dict:
                            worker.update_parameters(
                                weights=update_dict.get("weights"),
                                hyperparams=update_dict.get("hyperparams"),
                            )
                    except queue.Empty:
                        break

                # 2. Wait for work request
                cmd_args = {}
                if use_signaling and trigger_event is not None:
                    while not stop_flag.value:
                        if trigger_event.wait(timeout=0.1):
                            # Check if if there is a specific command for us in the queue
                            # (We use a simple name matching since all workers of this type share the event)
                            # However, in multi-process, only one worker might grab the command from the queue.
                            # So we peek or handle it carefully.
                            # Simplest: Each worker just tries to get its own command.
                            try:
                                # This is a bit tricky with multiple workers.
                                # For now, assume the orchestrator sends one command per worker or we broadcast.
                                # Given the project structure, we usually have one command per collection step.
                                pass
                            except:
                                pass
                            trigger_event.clear()
                            break

                    if stop_flag.value:
                        break

                # 3. Fetch task arguments if any
                while not command_queue.empty():
                    try:
                        name, args = command_queue.get_nowait()
                        if name == worker_cls.__name__:
                            cmd_args = args
                            break
                        else:
                            # Not for us, put it back (clunky but works for small num worker types)
                            command_queue.put((name, args))
                            time.sleep(0.01)
                    except queue.Empty:
                        break

                # 4. Dispatch Task
                if hasattr(worker, "collect"):
                    n = cmd_args.get("num_steps", 1000)
                    data = worker.collect(n)
                elif hasattr(worker, "evaluate"):
                    n = cmd_args.get("num_episodes", 1)
                    data = worker.evaluate(n)
                elif hasattr(worker, "reanalyze"):
                    n = cmd_args.get("batch_size", 32)
                    data = worker.reanalyze(n)
                elif hasattr(worker, "play_sequence"):
                    data = worker.play_sequence()
                else:
                    data = {}

                # Package result into standardized payload
                if isinstance(data, WorkerPayload):
                    payload = data
                elif isinstance(data, dict):
                    # Separate metrics (for logging) from data (for training) if necessary.
                    # For now, we assume dicts from standard actors are metrics.
                    payload = WorkerPayload(
                        worker_type=worker_cls.__name__, metrics=data
                    )
                else:
                    payload = WorkerPayload(
                        worker_type=worker_cls.__name__, metrics={}, data=data
                    )

                result_queue.put(payload)
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
        for i, (_, w) in enumerate(self.workers):
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

    def update_parameters(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        # 1. Update shared memory once in main process
        if weights and self.shared_networks:
            # Handle potentially compiled model state dicts
            clean_params = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
            for net in self.shared_networks:
                # This update is global; all workers automatically see it via shared RAM
                net.load_state_dict(clean_params, strict=False)

        # 2. Only send hyperparams to the workers to trigger local resets (e.g. noisy net)
        # We set weights=None for the IPC queue to avoid massive serialization overhead
        # (The Death Trap)
        update_dict = {"weights": None, "hyperparams": hyperparams or {}}

        # Send to all workers via the queue
        for _ in range(len(self.workers)):
            self.param_queue.put(update_dict)

    def request_work(self, worker_type: Type, **kwargs):
        """Signals the trigger event and sends arguments for the specified worker type."""
        type_name = worker_type.__name__
        # Broadcast command to all workers of this type
        # (Assuming all workers are idle and waiting for their own command)
        num_target_workers = len(
            [w for w_cls, w in self.workers if w_cls == worker_type]
        )
        # Just put one command per worker in the queue
        for _ in range(num_target_workers):
            self.command_queue.put((type_name, kwargs))

        if type_name in self.worker_events:
            self.worker_events[type_name].set()

    def stop(self):
        self.stop_flag.value = 1

        # Give workers a moment to see the stop flag
        time.sleep(0.1)

        for _, w in self.workers:
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
