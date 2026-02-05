---
trigger: always_on
---

---
trigger: always_on
---

# PyTorch Best Practices & Idioms

### Tensor Creation & Management
* **Factory Functions:** NEVER create a tensor on CPU and move it to GPU. Always use the `device` keyword argument in factory functions.
    * *Bad:* `torch.zeros((10, 10)).cuda()`
    * *Good:* `torch.zeros((10, 10), device=device)`
* **Avoid Deprecated Constructors:** NEVER use `torch.Tensor()` (uppercase). It allocates uninitialized memory. Use `torch.tensor()` (lowercase) for data or factory functions (`torch.rand`, `torch.ones`) for shapes.
* **No `.data` Access:** STRICTLY FORBIDDEN to use `tensor.data`. It bypasses autograd tracking and causes silent gradient bugs. Use `.detach()` if you need to stop gradients.
* **Modern Shape Ops:** Use `einops` for all complex tensor reshaping, permuting, and repeating. It makes code readable and shape-explicit.
    * *Example:* `rearrange(images, 'b c h w -> b (h w) c')` is clearer than `images.view(b, -1, c)`.

### Architecture Design
* **Module Lists:** NEVER use a Python `list` to store network layers. They will not be registered by `nn.Module` and their parameters will not be updated by the optimizer. Use `nn.ModuleList()` or `nn.Sequential()`.
* **Functional vs. Stateful:** Use `nn.functional` for stateless operations (activation functions like `F.relu`) unless they are part of an `nn.Sequential` block.
* **Evaluation Mode:** ALWAYS call `model.eval()` before validation/inference and `model.train()` before training. This affects Batch Norm and Dropout layers significantly.
### PyTorch Gradient Management
* **Default to `inference_mode`:** Always use `torch.inference_mode()` for inference loops, MCTS, and data generation. It disables view tracking and version counters, making it significantly faster and safer than `torch.no_grad()` for pure forward passes.
* **When to use `no_grad`:** Only fallback to `torch.no_grad()` if you are performing complex in-place tensor mutations that explicitly crash under `inference_mode` due to view-tracking limitations, or if the tensors created inside the context must be used in a later Autograd graph (rare in RL inference).
### Parameter Redundancy (Bias vs. Norm)
* **Rule:** If a Learnable Layer (`nn.Conv2d`, `nn.Linear`, `nn.ConvTranspose2d`) is immediately followed by a Normalization Layer (`nn.BatchNorm`, `nn.InstanceNorm`, `nn.GroupNorm`, `nn.LayerNorm`) that centers the data (subtracts mean), you MUST set `bias=False` on the learnable layer.
* **Reasoning:** The normalization layer subtracts the mean of the output, which mathematically cancels out any constant bias added by the previous layer. Keeping the bias creates a "dead" parameter that wastes memory and gradient computation.
* **Implementation Pattern:**
    ```python
    # Correct
    self.conv = nn.Conv2d(..., bias=False)
    self.norm = nn.BatchNorm2d(...)
    
    # Dynamic (if norm is optional)
    use_bias = (norm_type == "none")
    self.conv = nn.Conv2d(..., bias=use_bias)
    ```
### Gradient Clearing
* **Rule:** Always use `optimizer.zero_grad(set_to_none=True)` instead of `optimizer.zero_grad()`.
* **Reasoning:** Setting gradients to `None` instead of 0 avoids unnecessary memory operations (memset to 0) and allows the optimizer to skip the first addition (0 + grad) during backprop, treating it as a direct assignment instead.

### Kernel Fusion (Pointwise Operations)
* **Rule:** Explicitly apply `torch.compile` to "math-heavy" helper functions and Loss Pipelines, not just the neural network forward pass.
* **Why:** Operations like `clamp`, `exp`, `log`, `softmax`, and basic addition/multiplication are "memory-bound". Compiling them allows TorchInductor to fuse them into a single kernel, reading/writing memory only once instead of 10+ times.
* **Target Areas:**
    * Loss Pipelines (e.g., `loss_pipeline.run`)
    * Complex Activation Functions (if custom)
    * Return Estimators (GAE, N-Step calculations)
### Memory Format (Channels Last)
* **Rule:** When using `AMP` (Automatic Mixed Precision) with Convolutional Networks (`Conv2d`), you MUST convert the model and input tensors to `channels_last` memory format. ONLY IF THE DEVICE TYPE SUPPORTS THE OPERATION!
* **Why:** NVIDIA Tensor Cores are physically designed to multiply matrices in `NHWC` (Channels Last) format. If you use the default `NCHW` (Channels First), PyTorch has to secretly transpose your data back and forth for *every single convolution*, wasting massive amounts of VRAM bandwidth.
* **Implementation Pattern:**
    ```python
    # 1. Convert Model
    model = model.to(memory_format=torch.channels_last)
    
    # 2. Convert Inputs (in your training loop)
    input = input.to(device, memory_format=torch.channels_last)
    ```
### Production vs. Debug Mode
* **Rule:** Computationally expensive debugging tools (`autograd.detect_anomaly`, `profiler`, `gradcheck`) must be disabled by default and controlled via a centralized `debug` configuration flag.
* **Why:** `detect_anomaly` can slow down training by 300-500% because it stores the stack trace for every tensor operation. It should never be active during standard training runs.
### CPU Thread Affinity (OpenMP)
* **Rule:** Explicitly configure OpenMP environment variables inside CPU-heavy Ray Actors (Workers) before importing PyTorch.
* **Why:** Default PyTorch scheduling allows threads to "wander" across cores, causing cache thrashing and system stutter. Pinning threads (`OMP_PROC_BIND=CLOSE`) keeps data local to the core, improving MCTS speed.
* **Configuration:**
    * `OMP_NUM_THREADS`: Set to `num_allocated_cpus` (usually 1 or 2 per worker).
    * `OMP_PROC_BIND`: Set to `spread` or `close` to prevent OS jitter.
### Intel OpenMP Optimization
* **Rule:** When running on Intel CPUs, prioritize the Intel OpenMP Runtime (`libiomp`) over GNU OpenMP (`libgomp`).
* **Configuration:**
    * **Environment:** Set `KMP_BLOCKTIME=1` and `KMP_AFFINITY=granularity=fine,compact,1,0` to optimize for MCTS throughput.
    * **Dependency:** Prefer installing the `intel-openmp` PyPI package over manual `LD_PRELOAD` hacking.

### Automatic Mixed Precision (AMP)
* **Autocast Scope:** `torch.autocast` MUST wrap *only* the forward pass(es) and the loss computation.
* **Backward Exclusion:** DO NOT wrap the backward pass (`scaler.scale(loss).backward()`). Backward operations automatically run in the same `dtype` as their corresponding forward operations. Wrapping them is redundant and not recommended.

### Modularity & Reusability
* **Decoupled Components:** Architecture components (Loss functions, specific Blocks, Attention mechanisms) MUST be defined as standalone classes/functions, not hardcoded inside a massive `Agent` class.
* **Reuse Principle:** If a component (e.g., `ResBlock`) is used in more than one model, it must be extracted to a shared utility module.