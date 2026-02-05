---
name: tdd-workflow
description: Orchestrates a Test-Driven Development workflow for Machine Learning. It enforces writing tests before code, verifying failures, and ensuring >80% coverage on new features (Agents, Buffers, Custom Layers).
---

# Goal
Implement new features or refactor code using the Red-Green-Refactor cycle.

# Context
Use this skill when the user asks to "implement a new agent", "fix a bug", or "refactor" complex logic.

# Process (The Agent MUST follow this order)
1.  **Define the Journey:** Ask the user to define the "Researcher Journey" (e.g., *As a MuZero agent, I need to sample from a priority buffer so that I learn efficiently*).
2.  **Scaffold Test (Red):** Create a new test file in `tests/` using `pytest`.
    * **Constraint:** The test MUST fail initially (Red state).
    * **Focus:** Verify tensor shapes (`B, T, C, H, W`), NaN checks, and edge cases (empty buffers).
3.  **Run Test (Verify Failure):** Execute the test to confirm it fails for the right reason (using `scripts/run_tests.py`).
4.  **Implement (Green):** Write the minimum code required to pass the test.
    * **Constraint:** Use PyTorch best practices (vectorization over loops).
5.  **Verify (Green):** Run the test again to confirm it passes.
6.  **Refactor & Cover:**
    * Optimize the code (e.g., `torch.compile`, fusion).
    * Check coverage using `scripts/check_coverage.py`.
    * Ensure >80% coverage for the new module.

# Test Patterns
* **Unit:** Check input/output shapes of `nn.Module`.
* **Integration:** Run one step of `env.step(action)` with the Agent.
* **System:** "Overfit a single batch" loop to verify gradient flow.

# Examples
**User:** "Create a Prioritized Replay Buffer."
**Agent:** "Starting TDD workflow. Step 1: I will create `tests/replay_buffers/test_priority.py`. I'll add a test case to verify sampling probabilities match priorities. Running test now... (It failed as expected). Now implementing `PrioritizedReplayBuffer` class..."

**User:** "Fix the NaN error in the loss function."
**Agent:** "Step 1: I'll write a reproduction script `tests/repro_nan.py` that forces the specific input causing the NaN. Once I confirm it fails, I will add the epsilon stability fix."