# Modular RL Testing Suite

Reinforcement Learning bugs are rarely loud crashes; they are usually silent math errors—a detached gradient, a broadcast error, or a faulty advantage estimation—that manifest simply as "the agent isn't learning." 

This testing suite is designed to rigorously verify the underlying logic and mathematics of the codebase without requiring full training loops for every commit.

## 📁 Organization Philosophy

This directory uses a **hybrid organization strategy**:
1. **Unit Tests** strictly mirror the source code directory structure.
2. **Integration, Smoke, and Regression Tests** are organized by algorithm or feature.

```text
tests/
├── conftest.py                             # Global PyTorch fixtures (dummy tensors, seeded envs)
│
├── unit/                                   # Tier 1: strictly mirrors the source tree
│   ├── agents/                             #   ↳ e.g., tests/unit/agents/learner/functional/
│   ├── modules/                            #   ↳ e.g., tests/unit/modules/heads/
│   ├── replay_buffers/                     #   ↳ e.g., tests/unit/replay_buffers/
│   └── search/                             #   ↳ e.g., tests/unit/search/aos_search/
│
├── integration/                            # Tier 2: Organized by system interaction
│   ├── test_compilation.py                 #   ↳ Ensures torch.compile compatibility
│   └── test_loss_pipeline.py               #   ↳ Modules passing gradients correctly
│
├── smoke/                                  # Tier 3: Organized by execution target
│   ├── test_local_executor.py              #   ↳ Fast 2-epoch local loop
│   └── test_torch_mp_executor.py           #   ↳ Fast 2-epoch multiprocessing loop
│
└── regression/                             # Tier 4: Organized by algorithm/paper
    ├── ppo/                                #   ↳ Verifies PPO's 37 implementation details
    └── muzero/                             #   ↳ Verifies MuZero can solve tic-tac-toe
🚥 The 4-Tier Testing Pipeline
To maintain high developer velocity while ensuring mathematical rigor, tests are divided into four tiers based on execution time and determinism:

Tier 1: Unit (Core Math & Logic)
Scope: Isolated functions (Segment trees, GAE, MuZero MinMax stats, tensor shapes).

Execution: Runs in milliseconds.

CI Target: Runs on every single push.

Tier 2: Integration (Component Wiring)
Scope: Ensures modules wire together correctly without actual training (e.g., passing a dummy batch through the LossPipeline, testing Pufferlib environment adapters).

Execution: Runs in seconds.

CI Target: Runs on every Pull Request.

Tier 3: Smoke Tests (Execution Loops)
Scope: 1-to-2 epoch training loops on trivial environments (e.g., slippery_grid_world). Asserts that the loop runs, logs, and checkpoints without crashing, not that the agent learns.

Execution: Runs in minutes.

CI Target: Runs Nightly.

Tier 4: Algorithmic Regression (Feature Verification)
Scope: Full training jobs on baseline environments (e.g., CartPole) to ensure sample efficiency hasn't degraded and paper details (PPO, EfficientZero) remain intact.

Execution: Runs in tens of minutes to hours.

CI Target: Runs on Releases or Weekly.

## 🤝 Contract Testing: Protecting Architectural Assumptions

While pure math unit tests ensure the agent learns efficiently, **Contract Tests** ensure the agent doesn't crash 18 hours into a training run due to a shape mismatch or a broken dictionary payload.

In RL, modules act like microservices. Contract tests verify the strict formatting agreements between these modules (tensors, padding conventions, structural invariants) without validating the underlying math. These live inside `tests/unit/` (often named `test_[module]_contracts.py`) and execute in Tier 1.

### The 4 Core RL Contracts

When adding or modifying components, ensure the following contracts are explicitly tested:

**1. The Padding & Alignment Contract**
* **The Rule:** If a trajectory ends prematurely, the unrolled sequence must still equal length $K$. Padded steps must be zeroed, and the generated loss mask must explicitly ignore these indices during `.backward()`.
* **The Test:** Pass a sequence of length 2 into a module expecting length 5. Assert `shape == 5`, assert indices `[2:] == 0.0`, and assert the loss mask correctly identifies the invalid steps.

**2. The Terminal State Invariant**
* **The Rule:** The value of a terminal state is strictly `0.0`, regardless of neural network predictions.
* **The Test:** Pass a batch where `done = True`. Assert the bootstrap target values for those specific indices equal exactly `0.0`.

**3. The Shape and DType Contract**
* **The Rule:** PyTorch silently broadcasts mismatched tensors and promotes data types, causing silent memory bloat or massive logical errors.
* **The Test:** Isolate module boundaries (e.g., the transition between the Representation and Dynamics network). Explicitly `assert output.dtype == torch.float32` and `assert output.shape == (B, hidden_dim)`.

**4. The Dictionary/Payload Contract**
* **The Rule:** Distributed workers (Ray Actors) and Replay Buffers must agree on the exact keys of the transition payloads.
* **The Test:** Mock an environment step and assert `set(payload.keys()) == {"obs", "action", "reward", "done", "policy_logits"}`. Trigger extreme edge cases (like an empty dictionary) to ensure the buffer handles them gracefully.

⚖️ The 5 Rules of RL Test Driven Development
When adding tests to this repository, strictly adhere to the following rules:

The "Analytical Oracle" Rule: For math functions (GAE, n-step returns, UCB calculations), calculate the expected output by hand for a small sequence (e.g., length 3). Hardcode those exact float values into the test and use torch.testing.assert_close().

Strict Determinism: RL tests must be 100% reproducible. Rely on fixtures in conftest.py that strictly seed torch, numpy, and random before every test. Flaky tests are failing tests.

Explicit Tensor Shapes: A massive portion of PyTorch bugs stem from accidental broadcasting. Explicitly assert tensor.shape == expected_shape before asserting values.

The "Gradient Flow" Check: When testing complex pipelines, compute a dummy loss, call .backward(), and explicitly assert that .grad is not None for active networks, and is None for frozen/target networks.

Mock the Environment: Unit tests should never instantiate an actual Gym or Pufferlib environment. Pass synthetic PyTorch tensors (torch.randn(...)) to represent observations. Isolate agent logic from environment logic.

🚀 Getting Started
Run the fast unit tests locally before pushing:

Bash
pytest tests/unit/
Run tests with print statements for debugging:

Bash
pytest tests/unit/ -s