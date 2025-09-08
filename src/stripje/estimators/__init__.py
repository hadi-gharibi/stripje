"""
Estimators module - handlers for sklearn estimators (classifiers, regressors, etc.).
"""

# Import individual modules to register handlers
from . import ensemble, linear, naive_bayes, tree
from .ensemble import *

# Export handlers
from .linear import *
from .naive_bayes import *
from .tree import *

__all__ = linear.__all__ + tree.__all__ + ensemble.__all__ + naive_bayes.__all__
