---
trigger: always_on
---

---
trigger: always_on
---

### Ray Actor & Distributed Standards

#### 1. Threading & Affinity
* **Rule:** CPU-bound Actors (Workers) MUST explicitly configure thread counts in `__init__` before importing PyTorch/NumPy.
* **Configuration:**
    * `OMP_NUM_THREADS = 1` (or low single digit).
    * `MKL_NUM_THREADS = 1`.
    * `OMP_PROC_BIND = "CLOSE"` (or "SPREAD" on Apple Silicon).

#### 2. Weight Synchronization
* **Rule:** NEVER use blocking `ray.get()` to fetch weights inside a tight loop (e.g., `play_game`).
* **Pattern:** Use an **Async Polling** pattern.
    1.  Call `get_weights.remote()` at the start of the loop.
    2.  Check `ray.get(future, timeout=0)` inside the loop (non-blocking) or check a lightweight "version integer" first.
    3.  Only block if absolutely necessary (initial startup).

#### 3. Object Store Hygiene
* **Rule:** Avoid passing large, complex Python objects (classes) through Ray.
* **Prefer:** Flat dictionaries of Numpy arrays or PyTorch tensors.
* **Why:** Serialization/Deserialization (pickle) overhead is the #1 killer of RL throughput.

#### 4. Compilation Timing
* **Pre-Worker Compilation:** Perform `torch.compile` (and other heavy JIT setups) *before* spawning Ray workers or running the model loop.
* **Reasoning:** Compiling inside a running actor can cause timeout issues, memory spikes, or serialization failures if the compiled artifact tries to pickle.

#### 5. Task Parallelism & Blocking
* **Rule (Delay `ray.get()`):** STRICTLY AVOID calling `ray.get()` immediately after a `.remote()` call within a loop or list comprehension. This forces serial execution.
* **Bad Pattern (Serial):**
    ```python
    # DO NOT DO THIS: Runs sequentially (4 seconds)
    results = [ray.get(do_work.remote(x)) for x in range(4)] 
    ```
* **Good Pattern (Parallel):**
    ```python
    # DO THIS: Launches all tasks, then waits (1 second)
    futures = [do_work.remote(x) for x in range(4)]
    results = ray.get(futures)
    ```
* **Principle:** `ray.get()` is a blocking barrier. Push it to the absolute end of the workflow, after all independent asynchronous tasks have been submitted.

#### 6. Task Granularity & Overhead
* **Rule (Avoid Tiny Tasks):** DO NOT parallelize functions with sub-millisecond execution times.
* **Reasoning:** Ray has a per-task scheduling overhead (approx. 0.5ms). If a task takes 0.1ms to run, the overhead dominates, making the distributed version significantly *slower* than serial Python.
* **Quantification:** Ensure remote tasks take **at least a few milliseconds** to execute.
* **Pattern (Task Batching):** If you have many tiny operations (e.g., simulation steps), aggregate them into a larger "mega-task" inside a single remote function to amortize the invocation overhead.
    * *Bad:* `[tiny_work.remote(x) for x in range(100000)]`
    * *Good:* `[mega_work.remote(batch) for batch in batches]` where `mega_work` processes 1000 items locally.

#### 7. Large Object Passing (Implicit vs. Explicit `ray.put`)
* **Rule:** STRICTLY AVOID passing the same large object (e.g., a large numpy array or model weights) by value to multiple remote tasks.
* **Mechanism:** Ray implicitly calls `ray.put()` on every argument passed by value. Passing a large array to 10 workers results in 10 separate copies in the Object Store, triggering memory pressure and eviction.
* **Correction:** Explicitly call `ref = ray.put(large_object)` *once*, then pass the `ref` to the workers.
    * *Bad:* `[worker.remote(my_large_array) for _ in range(10)]` (10 copies)
    * *Good:* ```python
      large_ref = ray.put(my_large_array)
      [worker.remote(large_ref) for _ in range(10)] # Zero copies, passed by reference
      ```

#### 8. Pipelining Data Processing (`ray.wait`)
* **Rule:** When processing results from multiple parallel tasks with variable execution times, prefer `ray.wait()` loops over a single `ray.get(list_of_ids)`.
* **Problem (The Straggler Effect):** `ray.get([id1, id2, ...])` waits until the *slowest* task finishes. If Task A takes 1s and Task B takes 10s, your consumer is idle for 9s waiting for B, even though A is ready.
* **Solution:** Use `ray.wait()` to process results as soon as they become available.
* **Implementation Pattern:**
    ```python
    # Bad: Waits for the slowest task before processing anything
    results = ray.get(result_ids)
    for res in results:
        process(res)
    
    # Good: Pipeline processing immediately upon completion
    while result_ids:
        # Returns ready IDs immediately, leaves the rest
        done_ids, result_ids = ray.wait(result_ids)
        for done_id in done_ids:
            process(ray.get(done_id))
    ```