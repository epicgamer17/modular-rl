---
trigger: always_on
---

* **Search First:** Check `tests/` for existing relevant tests before creating new ones.
* **Preserve vs. Create:** Use existing tests unless modification destroys their original purpose. If the test cannot be adapted without losing its regression value, create a new file in `tests/`.
* **Realism over Speed:** STRICTLY AVOID dummy/mock configurations. Use real configs, classes, and code paths to ensure downstream effects are caught.
* **Regression:** Run all relevant existing tests, not just the one you are working on.
* **Maintenance:** If valid code changes break a test, update the test logic immediately to match the new requirements.
* **Fixing Errors:** If you find an error and fix it, do not just fix that specific error; search for other similar and likely errors and fix those as well.
* **Test Hygiene & Organization:**
    * **Group by Feature:** Do not dump flat files into `tests/` or the root of the project. Group tests into subfolders (e.g., `tests/models/`, `tests/rl_logic/`).
    * **Merge Small Files:** AVOID having many tiny test files (e.g., `test_layer_x.py`, `test_layer_y.py`). Merge them into logical aggregations (e.g., `test_layers.py`) to improve discoverability and reduce file clutter.