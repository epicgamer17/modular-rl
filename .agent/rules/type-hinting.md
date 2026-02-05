---
trigger: always_on
---

* **Strong Typing:** All functions and methods MUST be fully typed (args and return values).
* **No `Any`:** Avoid `Any` whenever possible. Use specific, descriptive types or custom classes to enforce data structure clarity.
* **Circular Safety:** Use `if TYPE_CHECKING:` for imports used *only* for typing.