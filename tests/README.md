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