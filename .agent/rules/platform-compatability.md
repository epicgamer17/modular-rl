---
trigger: always_on
---

---
trigger: always_on
---

### Cross-Platform Compatibility

#### 1. Hardware Guards
* **Rule:** Never assume `cuda` is available. Always use `torch.device` checks.
* **MPS (Mac) Support:**
    * Disable `torch.compile` on MPS (currently unsupported).
    * Use `torch.float16` for AMP on MPS (bfloat16 support is spotty).
* **Windows Support:**
    * Avoid `jax` or complex `subprocess` forks if possible.
    * Use `launcher.py` handles for environment variables instead of shell scripts.

#### 2. The Launcher Pattern
* **Rule:** Do not rely on `.sh` scripts for environment setup (jemalloc/OpenMP).
* **Standard:** All entry points must be runnable via `python launcher.py <command>` to ensure consistent memory allocation across OSs.

#### 3. Precision & Hardware Support
* **CPU Precision Rule:** STRICTLY AVOID `float16` on CPU. It requires slow emulation on most hardware.
* **CPU `bfloat16` Exception:** You MAY use `bfloat16` on CPUs that support it (Modern Intel/AMD with AVX-512/AMX or Apple Silicon M1/M2/M3).
    * *Implementation:* explicit check `if device.type == 'cpu' and not supports_bfloat16: use_float32`.