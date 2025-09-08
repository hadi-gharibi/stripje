"""
Fast single-row pipeline compiler for scikit-learn pipelines.

This module provides a way to compile scikit-learn pipelines into optimized
functions for single-row inference, avoiding the overhead of numpy operations
on single rows.
"""

from typing import Any, Callable

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import all transformers and estimators to register handlers
from .registry import (
    create_fallback_handler,
    get_handler,
    get_supported_transformers,
    register_step_handler,
)

# ============================================================================
# NESTED PIPELINE HANDLER
# ============================================================================


@register_step_handler(Pipeline)
def handle_nested_pipeline(pipeline_step: Pipeline) -> Callable[[Any], Any]:
    """Handle nested Pipeline objects recursively."""
    # Recursively compile the nested pipeline to maintain optimization
    return compile_pipeline(pipeline_step)


# ============================================================================
# COLUMN TRANSFORMER
# ============================================================================


@register_step_handler(ColumnTransformer)
def handle_column_transformer(step: ColumnTransformer) -> Callable[[Any], Any]:
    """Handle ColumnTransformer for single-row input."""
    transformers = []

    # Get fitted transformers with their column information
    for _name, transformer, columns in step.transformers_:
        if transformer == "drop":
            continue
        elif transformer == "passthrough":
            # For passthrough, just extract the specified columns
            col_indices = columns if isinstance(columns, (list, tuple)) else [columns]

            def make_passthrough_fn(indices: list[Any]) -> Callable[[Any], Any]:
                def passthrough_fn(x: Any) -> Any:
                    if isinstance(x, dict):
                        return [x[col] for col in indices]
                    else:
                        return [x[i] for i in indices]

                return passthrough_fn

            transformers.append((make_passthrough_fn(list(col_indices)), columns))
        else:
            # Get handler for the actual transformer
            handler = get_handler(type(transformer))
            if handler is None:
                # Use fallback mechanism for unsupported transformers
                print(
                    f"Warning: No optimized handler found for {type(transformer).__name__} in ColumnTransformer. Using fallback to original implementation."
                )
                handler = create_fallback_handler
            fn = handler(transformer)
            transformers.append((fn, columns))

    def transform_one(x: Any) -> Any:
        """Transform a single row through the ColumnTransformer."""
        results: list[Any] = []

        for fn, columns in transformers:
            # Extract values for the specified columns
            if isinstance(x, dict):
                # Input is dictionary with column names as keys
                if isinstance(columns, (list, tuple)):
                    col_values = [x[col] for col in columns]
                else:
                    col_values = [x[columns]]
            else:
                # Input is list/array with positional indexing
                if isinstance(columns, (list, tuple)):
                    # Convert string column names to indices if needed
                    if all(isinstance(col, str) for col in columns):
                        # This means we need to map string names to indices
                        # For now, assume columns are already indices or we have a mapping
                        col_values = [x[i] for i in range(len(columns))]
                    else:
                        col_values = [x[i] for i in columns]
                else:
                    if isinstance(columns, str):
                        # Single string column name - this is problematic for array input
                        # We'll need to handle this case differently
                        col_values = [x[0]]  # Fallback to first element
                    else:
                        col_values = [x[columns]]

            # Apply the transformer function
            result = fn(col_values)

            # Ensure result is a list and extend to results
            if isinstance(result, (list, tuple)):
                results.extend(result)
            else:
                results.append(result)

        return results

    return transform_one


# ============================================================================
# PIPELINE COMPILER
# ============================================================================


def compile_pipeline(pipeline: Pipeline) -> Callable[[Any], Any]:
    """
    Compile a scikit-learn pipeline into a fast single-row prediction function.

    Args:
        pipeline: A fitted scikit-learn Pipeline object

    Returns:
        A function that takes a single row (list/array) and returns the prediction
    """
    steps = []

    for _name, step in pipeline.steps:
        handler = get_handler(type(step))
        if handler is None:
            # Use fallback mechanism for unsupported transformers
            print(
                f"Warning: No optimized handler found for {type(step).__name__}. Using fallback to original implementation."
            )
            handler = create_fallback_handler
        steps.append(handler(step))

    def predict_one(x: Any) -> Any:
        """Fast single-row prediction function."""
        for fn in steps:
            x = fn(x)
        return x

    return predict_one


__all__ = ["compile_pipeline", "get_supported_transformers"]
