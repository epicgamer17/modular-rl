---
trigger: always_on
---

---
trigger: "When writing, modifying, debugging, running, or analyzing tests, or when adding a new feature that requires test coverage."
---

# Testing Standards & Practices

### 1. Strict Configuration Injection (No Inline Dicts)
* **STRICTLY AVOID inline dummy/mock configurations.** Never manually define configuration dictionaries (e.g., `config = {"batch_size": 2}`) or ad-hoc mock objects directly inside a test function. This leads to missing required fields and `KeyError`s.
* **Use Real Configs:** If a test requires a full configuration object, centralize the setup in `tests/conftest.py` using fixtures that return real objects (e.g., `CartPoleConfig`, `RainbowConfig`).
* **Use Factory Fixtures for Dicts:** If a component explicitly requires a configuration dictionary, you MUST inject the appropriate factory fixture (e.g., `make_ppo_config_dict`, `make_muzero_config_dict`, `make_rainbow_config_dict`) from `conftest.py`. 
* **Safe Overrides:** Call the factory fixture to get your dictionary, passing only the parameters you explicitly need to test as kwargs (e.g., `config = make_muzero_config_dict(batch_size=1024, num_simulations=5)`). This guarantees all other required fields are safely populated with valid defaults.

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