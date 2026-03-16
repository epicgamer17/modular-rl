---
name: run-smoke-test
description: Run a quick smoke test to verify the RL framework is working end-to-end. Use when checking basic functionality after changes or before committing.
---

# Run Smoke Test

Quick end-to-end sanity check for the RL framework.

## Steps

1. Set `ANTIGRAVITY_AGENT=1` (required for torch.compile artifact isolation).
2. Run a minimal training loop (1-2 steps) on CartPole or another fast env.
3. Check for NaN losses, shape errors, or import failures.
4. Report pass/fail.

## Example

```bash
ANTIGRAVITY_AGENT=1 python -m pytest tests/ -m "not slow" -x -q
```
