"""Backend router for search implementations.

This package routes imports to either:
- Python implementation in ``search_py/`` (default)
- C++ pybind11 module ``rainbow_search_cpp`` when requested
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent
_SEARCH_PY_DIR = _PACKAGE_DIR.parent / "search_py"
_VALID_BACKENDS = {"python", "cpp"}

_active_backend: str | None = None
_cpp_module: Any = None


def _extract_backend_name(config: Any = None) -> str | None:
    if config is None:
        return None

    if isinstance(config, str):
        return config

    if isinstance(config, Mapping):
        value = config.get("search_backend")
        return None if value is None else str(value)

    value = getattr(config, "search_backend", None)
    return None if value is None else str(value)


def _resolve_backend_name(config: Any = None) -> str:
    backend = _extract_backend_name(config) or os.getenv("SEARCH_BACKEND", "python")
    backend = str(backend).strip().lower()
    if backend not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Unsupported search backend {backend!r}. Expected one of: {valid}.")
    return backend


def _clear_search_submodules() -> None:
    prefix = f"{__name__}."
    for module_name in list(sys.modules):
        if module_name.startswith(prefix):
            sys.modules.pop(module_name, None)


def _activate_python_backend() -> None:
    if not _SEARCH_PY_DIR.is_dir():
        raise ImportError(f"Python backend directory not found: {_SEARCH_PY_DIR}")
    __path__[:] = [str(_SEARCH_PY_DIR)]


def _activate_cpp_backend() -> Any:
    return importlib.import_module("rainbow_search_cpp")


def configure_backend(config: Any = None) -> str:
    """Configure backend from config object/dict/string and route imports.

    Accepted forms:
    - ``"cpp"`` / ``"python"``
    - ``{"search_backend": "cpp"}``
    - ``config.search_backend == "cpp"``
    """

    global _active_backend
    global _cpp_module

    requested_backend = _resolve_backend_name(config)
    if _active_backend == requested_backend:
        return _active_backend

    _clear_search_submodules()
    _cpp_module = None

    if requested_backend == "cpp":
        try:
            _cpp_module = _activate_cpp_backend()
            __path__[:] = []
            _active_backend = "cpp"
            return _active_backend
        except ImportError as exc:
            warnings.warn(
                f"C++ search backend requested but unavailable ({exc}). Falling back to Python backend.",
                RuntimeWarning,
                stacklevel=2,
            )

    _activate_python_backend()
    _active_backend = "python"
    return _active_backend


def get_backend_name() -> str:
    if _active_backend is None:
        return configure_backend()
    return _active_backend


def __getattr__(name: str) -> Any:
    if get_backend_name() == "cpp" and _cpp_module is not None and hasattr(_cpp_module, name):
        return getattr(_cpp_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    if get_backend_name() == "cpp" and _cpp_module is not None:
        names.update(dir(_cpp_module))
    return sorted(names)


# Alias names for convenience
configure = configure_backend
set_backend = configure_backend

# Activate backend at import time (default: python)
configure_backend()
