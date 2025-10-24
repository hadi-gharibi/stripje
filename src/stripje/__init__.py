"""
Sklean - Make sklearn pipelines lean and efficient with fast single-row inference compilation.
"""

# Import all handler modules to register them
from . import estimators, transformers
from .__version__ import __version__
from .fast_pipeline import compile_pipeline, get_supported_transformers

__all__ = [
    "__version__",
    # Core compilation
    "compile_pipeline",
    "get_supported_transformers",
]
