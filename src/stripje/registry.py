"""
Registry for step handlers.
"""

from typing import Any, Callable, Optional, TypeVar

import numpy as np

# Type variable for sklearn estimators/transformers
T = TypeVar("T")

# Registry for step handlers
STEP_HANDLERS: dict[type, Callable[[Any], Callable]] = {}


def register_step_handler(
    step_type: type[T],
) -> Callable[[Callable[[T], Callable]], Callable[[T], Callable]]:
    """Decorator to register a handler for a specific step type."""

    def decorator(fn: Callable[[T], Callable]) -> Callable[[T], Callable]:
        STEP_HANDLERS[step_type] = fn
        return fn

    return decorator


def get_handler(step_type: type) -> Optional[Callable]:
    """Get handler for a specific step type."""
    return STEP_HANDLERS.get(step_type)


def create_fallback_handler(step: Any) -> Callable:
    """
    Create a fallback handler that uses the original step for unsupported transformers.

    This handler wraps the original step's transform or predict method to work with
    single-row inputs while maintaining the same interface as optimized handlers.
    """

    def fallback_fn(x: Any) -> Any:
        """Fallback function that uses the original step."""
        # Convert single row to numpy array with proper shape
        if hasattr(step, "transform"):
            # This is a transformer
            x_array = np.array([x])  # Add batch dimension
            result = step.transform(x_array)
            # Return as list/single value depending on output shape
            if result.shape[1] == 1:
                return result[0, 0]  # Single value output
            else:
                return result[0].tolist()  # Multiple values as list
        elif hasattr(step, "predict"):
            # This is an estimator
            x_array = np.array([x])  # Add batch dimension
            result = step.predict(x_array)
            return result[0]  # Return single prediction
        else:
            raise ValueError(
                f"Step {type(step).__name__} has neither transform nor predict method"
            )

    return fallback_fn


def get_supported_transformers() -> list[type]:
    """Return a list of supported transformer types."""
    return list(STEP_HANDLERS.keys())


__all__ = [
    "register_step_handler",
    "get_handler",
    "get_supported_transformers",
    "create_fallback_handler",
    "STEP_HANDLERS",
]
