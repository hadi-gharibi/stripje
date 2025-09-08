"""
Transformers module - handlers for sklearn preprocessing, feature selection, and decomposition.
"""

# Import individual modules to register handlers
from . import decomposition, feature_selection, preprocessing

# Import contrib modules (optional dependencies)
try:
    from . import contrib

    contrib_all = contrib.__all__
except ImportError:
    contrib_all = []

# Export commonly used handlers
from .decomposition import *
from .feature_selection import *
from .preprocessing import *

__all__ = (
    preprocessing.__all__
    + feature_selection.__all__
    + decomposition.__all__
    + contrib_all
)
