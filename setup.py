from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


CPP_DIR = Path("search") / "search_cpp"
CPP_SOURCES = [
    "bindings.cpp",
    "backprop.cpp",
    "min_max_stats.cpp",
    "modular_search.cpp",
    "nodes.cpp",
    "scoring.cpp",
    "selection.cpp",
]

ext_modules = [
    Pybind11Extension(
        "mcts_cpp_backend",
        [str(CPP_DIR / src) for src in CPP_SOURCES],
        include_dirs=[str(CPP_DIR)],
        cxx_std=17,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
