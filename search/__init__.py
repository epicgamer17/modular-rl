"""Dynamic facade for selecting the search backend.

Backend selection order:
1) Explicit call to ``configure_backend(...)``
2) ``config.backend`` / ``config.search_backend`` (or dict equivalents)
3) ``MCTS_BACKEND`` environment variable
4) Default: ``python``
"""

from __future__ import annotations

import importlib
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent
_SEARCH_PY_DIR = _PACKAGE_DIR / "search_py"
_VALID_BACKENDS = {"python", "cpp"}
_DEFAULT_BACKEND = "python"
_BACKEND_ENV_VAR = "MCTS_BACKEND"

# Keep python submodule imports working: `from search.search_factories import ...`
__path__[:] = [str(_PACKAGE_DIR), str(_SEARCH_PY_DIR)]

_active_backend: str | None = None
_cpp_module: Any = None


def _extract_backend_from_config(config: Any) -> str | None:
    if config is None:
        return None
    if isinstance(config, str):
        return config
    if isinstance(config, Mapping):
        if "backend" in config:
            return str(config["backend"])
        if "search_backend" in config:
            return str(config["search_backend"])
        search_block = config.get("search")
        if isinstance(search_block, Mapping):
            if "backend" in search_block:
                return str(search_block["backend"])
            if "search_backend" in search_block:
                return str(search_block["search_backend"])
        return None

    # Object-style config
    value = getattr(config, "backend", None)
    if value is None:
        value = getattr(config, "search_backend", None)
    if value is not None:
        return str(value)

    search_obj = getattr(config, "search", None)
    if search_obj is not None:
        if isinstance(search_obj, Mapping):
            if "backend" in search_obj:
                return str(search_obj["backend"])
            if "search_backend" in search_obj:
                return str(search_obj["search_backend"])
        else:
            value = getattr(search_obj, "backend", None)
            if value is None:
                value = getattr(search_obj, "search_backend", None)
            if value is not None:
                return str(value)
    return None


def _resolve_backend_name(config: Any = None, backend: str | None = None) -> str:
    candidate = backend or _extract_backend_from_config(config)
    if candidate is None:
        candidate = os.getenv(_BACKEND_ENV_VAR, _DEFAULT_BACKEND)
    resolved = str(candidate).strip().lower()
    if resolved not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Unknown search backend {resolved!r}. Expected one of: {valid}.")
    return resolved


def _load_cpp_backend_module():
    module_candidates = [
        os.getenv("MCTS_CPP_MODULE", "mcts_cpp_backend"),
        "search_cpp",
        "rainbow_search_cpp",
    ]
    last_exc: Exception | None = None
    for module_name in module_candidates:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_exc = exc
    if last_exc is None:
        raise ImportError("Could not locate any C++ search backend module.")
    raise ImportError(str(last_exc))


def configure_backend(config: Any = None, backend: str | None = None) -> str:
    """Configure the active search backend."""

    global SearchAlgorithm
    global MinMaxStats
    global _active_backend
    global _cpp_module

    requested = _resolve_backend_name(config=config, backend=backend)
    if requested == _active_backend:
        return requested

    if requested == "cpp":
        try:
            module = _load_cpp_backend_module()
            SearchAlgorithm = module.SearchAlgorithm
            MinMaxStats = module.MinMaxStats
            _cpp_module = module
            _active_backend = "cpp"
            return requested
        except Exception as exc:
            warnings.warn(
                f"C++ backend requested but unavailable ({exc}). Falling back to Python backend.",
                RuntimeWarning,
                stacklevel=2,
            )

    from .search_py.modular_search import SearchAlgorithm as _PySearchAlgorithm
    from .search_py.min_max_stats import MinMaxStats as _PyMinMaxStats

    SearchAlgorithm = _PySearchAlgorithm
    MinMaxStats = _PyMinMaxStats
    _cpp_module = None
    _active_backend = "python"
    return _active_backend


def set_backend(backend: str) -> str:
    return configure_backend(backend=backend)


def get_backend_name() -> str:
    if _active_backend is None:
        return _resolve_backend_name()
    return _active_backend


def __getattr__(name: str) -> Any:
    if name in {"SearchAlgorithm", "MinMaxStats"}:
        if _active_backend is None:
            configure_backend()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SearchAlgorithm",
    "MinMaxStats",
    "configure_backend",
    "set_backend",
    "get_backend_name",
]
