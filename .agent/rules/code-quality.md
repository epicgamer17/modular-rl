---
trigger: always_on
---

* **No Magic Values:** Define numbers/strings as CONSTANTS at the top of the file.

* **Import Hygiene:** STRICTLY FORBID `sys.path.append(...)` hacks to resolve local modules.
    * *Solution:* Structure the project as a proper Python package (use `setup.py` or `pyproject.toml`) or run as a module (`python -m my_package.main`).
* **OOP Best Practices:**
    * **Inheritance:** Use inheritance to define shared behavior (e.g., `BaseAgent`, `BaseConfig`).
    * **Composition over Duplication:** If two classes share significant logic but aren't strictly parent-child, extract the shared logic into a helper class or mixin.
    * **Modularity:** Design components (Losses, Networks, Loggers) to be pluggable. Avoid tight coupling where Class A instantiates Class B directly; inject Class B as a dependency instead.