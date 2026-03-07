---
trigger: always_on
---

---
trigger: "When writing, modifying, debugging, running, or analyzing tests, or when adding a new feature that requires test coverage."
---

# Testing Standards & Practices

### 1. The "No Mocks" Rule (Real Code Paths Only)
* **STRICTLY AVOID dummy/mock configurations.** Never use `MagicMock`, fake dictionaries, or ad-hoc mock objects for game configs or network parameters.
* **Use Real Configs:** Centralize configuration setups in `tests/conftest.py` using fixtures that return real objects (e.g., `CartPoleConfig`, `RainbowConfig`).
* **Factory Fixtures:** For tests needing slightly altered settings, inject the factory fixture (e.g., `make_cartpole_config(**overrides)`) which safely returns a deep copy of a base config with your overrides, ensuring no global state mutation.

### 2. Strict Pytest Markers (MANDATORY)
* Every single test file MUST declare a module-level pytest marker at the very top of the file, just below the imports. 
* **Syntax:** `pytestmark = pytest.mark.<category>`
* **Categories:**
  * `unit`: Fast, isolated tests (e.g., pure math, shapes, logic, utility checks).
  * `integration`: Component interaction tests (e.g., passing network output into a search tree).
  * `slow`: Full training loops, batched environments, or heavy rollouts.
  * `regression`: Automated tests built to reproduce and guard against previously fixed bugs.
* **NEVER** write a test file without one of these markers.

### 3. File Organization & Naming
* **Group by Feature:** All tests must reside in specific architectural subdirectories inside `tests/` (e.g., `tests/search/`, `tests/replay_buffers/`, `tests/modules/`, `tests/agents/`). Do not put standard tests in the root `tests/` directory.
* **Naming Convention:** All test files must strictly follow the `test_<component>_<behavior>.py` format.
* **Regression Tests:** Any script designed to reproduce a bug must be placed in `tests/regression/`, renamed to `test_regression_<issue>.py`, and wrapped in a standard pytest function with strict `assert` checks.

### 4. Pure Pytest & State Isolation
* **No `unittest`:** Do not use `unittest.TestCase` classes. Write standalone, flat test functions using standard Python `assert a == b`.
* **No Global Mutations:** Do not mutate class attributes globally to set up a test. If specific attributes need changing, use Pytest's built-in `monkeypatch` fixture to safely and temporarily override them.
* **No `if __name__ == "__main__":`:** STRICTLY FORBIDDEN to include main blocks in test files. Tests should only be run via the `pytest` CLI.
* **Determinism:** For tests with stochastic elements (e.g., policy distributions, MCTS), enforce strict determinism by explicitly setting `torch.manual_seed(42)` and `np.random.seed(42)`. Assert exact tensor values using `torch.allclose()`.

### 5. Search First & Preserve vs. Create
* **Search First:** Always check the `tests/` directory for existing, relevant tests before creating a brand new file.
* **Preserve vs. Create:** When modernizing an out-of-date test, attempt to fix its inputs using centralized `conftest.py` fixtures. If updating it completely destroys its original purpose, delete it and write a new one targeting the modern codebase.

### 6. Coverage Quality
* Focus on testing logic branches by passing minimal, deterministic configurations (e.g., a tiny grid-world game, search depth of 2, small batch sizes).
* **Test the "Unhappy Path":** Explicitly pass invalid input shapes or illegal configurations, and use `pytest.raises(ExpectedException)` to verify the code crashes safely.
* *Note: Any code inside `deprecated/` folders is excluded from test coverage.*

---

**🚨 COMPLIANCE CHECKLIST (You must satisfy all before finishing any test generation/modification):**
- [ ] Did I avoid using `MagicMock` or fake dictionaries?
- [ ] Did I use a real, lightweight configuration injected from `conftest.py`?
- [ ] Did I add the module-level marker (e.g., `pytestmark = pytest.mark.unit`) at the very top of the file?
- [ ] Did I set `torch.manual_seed(42)` and `np.random.seed(42)` if the test involves sampling or neural networks?
- [ ] Is the test using pure `pytest` functions (no `unittest.TestCase`) and standard `assert` statements?
- [ ] Did I ensure NO `if __name__ == "__main__":` block exists in the test file?