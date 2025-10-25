"""
Estimators module - handlers for sklearn estimators (classifiers, regressors, etc.).
"""

import importlib
import pkgutil
from pathlib import Path

# Dynamically discover and import all modules in this package
_current_dir = Path(__file__).parent

# Import all Python modules in the current directory (except __init__.py)
for module_info in pkgutil.iter_modules([str(_current_dir)]):
    if not module_info.name.startswith("_"):
        importlib.import_module(f".{module_info.name}", package=__name__)

# Dynamically build __all__ from all imported modules
__all__ = []
for name in dir():
    if not name.startswith("_"):
        obj = globals()[name]
        # Include functions and classes, but exclude modules and imported utilities
        if callable(obj) and not isinstance(obj, type(pkgutil)):
            __all__.append(name)
