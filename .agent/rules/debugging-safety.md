---
trigger: always_on
---

---
trigger: always_on
---

# Debugging & Safety Standards

### Shape & Type Safety
* **Shape Comments/Typing:** If not using `einops`, every complex tensor operation MUST have a comment explaining the shape transformation: `# [B, C, H, W] -> [B, H*W, C]`.
* **Jaxtyping (Recommended):** Use `jaxtyping` or strict assertions to enforce shapes at function boundaries.
    * *Example:* `def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch classes"]:`
* **Assert Device:** When accepting tensors in a Module, assert they are on the expected device if mixed-device logic exists.

### Reproducibility
* **Seeding:** Reproducibility is critical for RL debugging. Set seeds for all backends at the entry point:
    ```python
    import random, numpy, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ```

### Error Handling
* **NaN Checks:** In RL, "exploding gradients" often appear as NaNs. If loss becomes NaN, use `torch.autograd.detect_anomaly()` temporarily to find the source, but remove it in production (it slows down training).

* **Disabling torch.compile** Always disable torch.compile to check if the code runs correctly.

