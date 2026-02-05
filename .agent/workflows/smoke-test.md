---
description: Runs a short, lightweight training loop to verify configuration validity.
---

# Goal
Ensure the current agent configuration and network architecture are valid by running a minimal training loop.

# Steps
1. **Identify Config**
   - Ask the user which config to test (default to `agent_configs/muzero_config.py` if unspecified).

2. **Run Training Wrapper**
   - Execute the training script with flags for a minimal run:
     ```bash
     python launcher.py python -m agents.muzero --config <config_name> --debug --smoke-test --steps 50
     ```
     *(Note: Ensure your `muzero.py` supports a `--smoke-test` or similar flag that forces a tiny batch size and 1 epoch).*

3. **Verify**
   - Check logs for "Training Step 50 complete".
   - Check for common errors: `RuntimeError: shape mismatch`, `CUDA out of memory`.

4. **Outcome**
   - If successful, output: "✅ Config is valid. Ready for full experiment."