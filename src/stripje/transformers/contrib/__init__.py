"""
Contrib transformers module - handlers for third-party transformers like category_encoders.
"""

# Import category encoders if available
try:
    from . import category_encoders_transformers
    from .category_encoders_transformers import *

    __all__ = category_encoders_transformers.__all__
except ImportError:
    # category_encoders not installed
    __all__ = []
