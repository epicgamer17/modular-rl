---
trigger: always_on
---

* **Fail Fast:** MINIMIZE `try-except` blocks. Never suppress errors to keep the program running.
* **Use Asserts:** Place `assert` statements at the start (preconditions) and end (postconditions) of functions to verify assumptions.
* **Descriptive Messages:** Every `assert` MUST include a clear text message explaining the failure (e.g., `assert x > 0, "Value x must be positive, got {x}"`).
* **Strict Attributes:** DO NOT use `getattr(obj, 'attr', default)`. Use `assert hasattr(obj, 'attr'), "Missing attr"` followed by direct access.
