---
trigger: always_on
---

---
trigger: always_on
---

# Performance & Optimization Rules

### Computation Graph
* **Inference Mode:** Use `with torch.inference_mode():` for all actor/inference loops (MCTS implementation). It is faster than `torch.no_grad()` as it disables view-tracking and version counters.
* **Detach Memory:** NEVER store attached tensors in a Replay Buffer. ALWAYS call `.detach().cpu()` before appending. Failing to do so keeps the entire computation graph alive, causing massive memory leaks.
* **Scalar Logging:** Use `.item()` when logging metrics (loss, reward). Storing 0-d tensors keeps graph history attached.

### GPU & Compilation
* **Torch Compile:** Use `torch.compile(model)` for the main network.
    * *Constraint:* Ensure input tensor shapes are consistent (e.g., consistent batch sizes) to avoid expensive recompilations.
    * *Constraint:* Avoid "graph breaks" like printing tensors or using numpy inside the forward pass.
* **Lazy GPU Transfer:** Do not push Replay Buffers to GPU. Keep data on CPU and move to GPU *only* immediately before the forward pass/training step.
* **Avoid Loops:** STRICTLY AVOID iterating over tensor dimensions in Python (e.g., `for i in range(batch):`). PyTorch is optimized for vectorized operations; Python loops kill performance.

### Compilation & Mixed Precision
* **Minimize Graph Breaks:** When using `torch.compile`, actively monitor and eliminate graph breaks (e.g., passing tensors to external libraries, data-dependent control flow). Breaks force Python fallback, negating speedups.
* **AMP + Compile:** ALWAYS combine `torch.compile` with Automatic Mixed Precision (AMP) where possible. The inductor backend is specifically optimized to fuse kernels for mixed-precision arithmetic.
* **Find Breaks Explicitly:** Use `torch.compile(fullgraph=self.config.fullgraph)` during development/debugging to strictly enforce a single graph. This will raise an error if a graph break occurs, identifying the exact line causing the fallback.
    * *Note:* Remove `fullgraph=self.config.fullgraph` in production if some breaks are unavoidable, but strive to eliminate them first.