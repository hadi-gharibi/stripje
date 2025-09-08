"""
Profiling utilities for sklearn pipelines and compiled pipelines.

This module provides comprehensive profiling capabilities for both standard
sklearn pipelines and compiled pipelines with performance comparisons.
"""

from __future__ import annotations

import copy
import gc
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .fast_pipeline import compile_pipeline

__all__ = [
    "ProfileResult",
    "PipelineProfiler",
    "CompiledPipelineProfiler",
    "CompiledProfileResult",
    "profile_pipeline_compilation",
]


@dataclass
class ProfileResult:
    """Container for profiling results of a single step."""

    step_name: str
    operation: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    num_runs: int
    step_type: str
    children: list[ProfileResult] = field(default_factory=list)

    def _format_time(self, time_seconds: float) -> tuple[float, str]:
        """
        Format time with appropriate unit (microseconds, milliseconds, seconds, minutes).

        Returns:
            tuple: (formatted_time, unit)
        """
        if (
            np.isnan(time_seconds)
            or time_seconds == float("inf")
            or time_seconds == float("-inf")
        ):
            return time_seconds, "s"

        if time_seconds < 1e-3:  # Less than 1 millisecond
            return time_seconds * 1e6, "μs"
        elif time_seconds < 1.0:  # Less than 1 second
            return time_seconds * 1e3, "ms"
        elif time_seconds < 60.0:  # Less than 1 minute
            return time_seconds, "s"
        else:  # 1 minute or more
            return time_seconds / 60.0, "min"

    def get_formatted_times(self) -> dict[str, str]:
        """Get all times formatted with appropriate units."""
        mean_val, mean_unit = self._format_time(self.mean_time)
        std_val, std_unit = self._format_time(self.std_time)
        min_val, min_unit = self._format_time(self.min_time)
        max_val, max_unit = self._format_time(self.max_time)

        return {
            "mean": f"{mean_val:.3f}{mean_unit}",
            "std": f"{std_val:.3f}{std_unit}",
            "min": f"{min_val:.3f}{min_unit}",
            "max": f"{max_val:.3f}{max_unit}",
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        formatted_times = self.get_formatted_times()
        return {
            "step_name": self.step_name,
            "operation": self.operation,
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "mean_time_formatted": formatted_times["mean"],
            "std_time_formatted": formatted_times["std"],
            "min_time_formatted": formatted_times["min"],
            "max_time_formatted": formatted_times["max"],
            "num_runs": self.num_runs,
            "step_type": self.step_type,
            "children": [child.to_dict() for child in self.children],
        }


class PipelineProfiler:
    """
    A comprehensive profiler for sklearn pipelines that can measure timing
    for various operations including nested pipelines and ColumnTransformers.

    Features:
    - Hierarchical profiling of nested pipelines and ColumnTransformers
    - Support for fit, predict, transform, and fit_transform operations
    - Robust single-row performance testing with cache invalidation
    - Statistical analysis over multiple runs
    - Detailed reporting with timing breakdown
    """

    def __init__(
        self,
        warmup_runs: int = 3,
        profile_runs: int = 10,
        cache_invalidation: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the profiler.

        Parameters:
        -----------
        warmup_runs : int, default=3
            Number of warmup runs to perform before actual profiling
        profile_runs : int, default=10
            Number of runs to perform for timing measurements
        cache_invalidation : bool, default=True
            Whether to perform cache invalidation between runs
        verbose : bool, default=False
            Whether to print progress information
        """
        self.warmup_runs = warmup_runs
        self.profile_runs = profile_runs
        self.cache_invalidation = cache_invalidation
        self.verbose = verbose
        self.results: dict[str, ProfileResult] = {}

    def _invalidate_caches(self) -> None:
        """Invalidate CPU caches and trigger garbage collection."""
        if self.cache_invalidation:
            # Force garbage collection
            gc.collect()

            # Create some dummy operations to flush CPU cache
            dummy_array = np.random.random((1000, 100))
            _ = np.sum(dummy_array)
            del dummy_array
            gc.collect()

    def _time_operation(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> list[float]:
        """
        Time an operation multiple times with cache invalidation.

        Parameters:
        -----------
        operation : Callable
            The operation to time
        *args, **kwargs
            Arguments to pass to the operation

        Returns:
        --------
        List[float]
            List of timing measurements
        """
        times = []

        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                operation(*args, **kwargs)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning during warmup: {e}")
            self._invalidate_caches()

        # Actual timing runs
        for i in range(self.profile_runs):
            if self.verbose and i % max(1, self.profile_runs // 5) == 0:
                print(f"  Run {i + 1}/{self.profile_runs}")

            self._invalidate_caches()

            start_time = time.perf_counter()
            try:
                operation(*args, **kwargs)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # Sanity check for timing
                if elapsed_time >= 0 and elapsed_time < 3600:  # Less than 1 hour
                    times.append(elapsed_time)
                elif self.verbose:
                    print(f"  Warning: Suspicious timing {elapsed_time:.6f}s, skipping")

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Operation failed: {e}")
                # Don't add inf times, just skip failed runs

        return times

    def _get_step_type(self, step: Any) -> str:
        """Get the type of a pipeline step."""
        if isinstance(step, Pipeline):
            return "Pipeline"
        elif isinstance(step, ColumnTransformer):
            return "ColumnTransformer"
        elif hasattr(step, "__class__"):
            return str(step.__class__.__name__)
        else:
            return str(type(step).__name__)

    def _profile_step(
        self,
        step: Any,
        step_name: str,
        operation: str,
        X: Any,
        y: Any | None = None,
        is_fitted: bool = False,
    ) -> ProfileResult:
        """
        Profile a single step in the pipeline.

        Parameters:
        -----------
        step : Any
            The pipeline step to profile
        step_name : str
            Name of the step
        operation : str
            Operation to profile ('fit', 'transform', 'predict', 'fit_transform')
        X : Any
            Input data
        y : Any, optional
            Target data for supervised operations
        is_fitted : bool, default=False
            Whether the step is already fitted

        Returns:
        --------
        ProfileResult
            Profiling results for this step
        """
        step_type = self._get_step_type(step)

        if self.verbose:
            print(f"Profiling {step_name} ({step_type}) - {operation}")

        # Check if operation is supported
        if operation == "fit" and not hasattr(step, "fit"):
            return self._create_unsupported_result(
                step_name, operation, step_type, "No fit method"
            )
        elif operation == "transform" and not hasattr(step, "transform"):
            return self._create_unsupported_result(
                step_name, operation, step_type, "No transform method"
            )
        elif operation == "predict" and not hasattr(step, "predict"):
            return self._create_unsupported_result(
                step_name, operation, step_type, "No predict method"
            )
        elif operation == "fit_transform" and not hasattr(step, "fit_transform"):
            return self._create_unsupported_result(
                step_name, operation, step_type, "No fit_transform method"
            )

        # For predict operations, ensure the step is fitted
        if (
            operation in ["predict", "transform"]
            and not is_fitted
            and hasattr(step, "fit")
        ):
            try:
                if self.verbose:
                    print(f"  Fitting {step_name} before {operation}")
                step.fit(X, y)
                is_fitted = True
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not fit {step_name}: {e}")
                return self._create_failed_result(
                    step_name, operation, step_type, f"Fit failed: {e}"
                )

        # Determine the operation to perform
        if operation == "fit":

            def op_func() -> Any:
                return step.fit(X, y)
        elif operation == "transform":

            def op_func() -> Any:
                return step.transform(X)
        elif operation == "predict":

            def op_func() -> Any:
                return step.predict(X)
        elif operation == "fit_transform":

            def op_func() -> Any:
                return step.fit_transform(X, y)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        # Time the operation
        times = self._time_operation(op_func)

        if not times:
            # All runs failed
            return self._create_failed_result(
                step_name, operation, step_type, "All timing runs failed"
            )

        # Calculate statistics
        times_array = np.array(times)
        result = ProfileResult(
            step_name=step_name,
            operation=operation,
            mean_time=float(np.mean(times_array)),
            std_time=float(np.std(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            num_runs=len(times),
            step_type=step_type,
        )

        # Profile children for composite steps
        if isinstance(step, Pipeline):
            result.children = self._profile_pipeline_steps(
                step, operation, X, y, is_fitted
            )
        elif isinstance(step, ColumnTransformer):
            result.children = self._profile_column_transformer_steps(
                step, operation, X, y, is_fitted
            )

        return result

    def _create_unsupported_result(
        self, step_name: str, operation: str, step_type: str, reason: str
    ) -> ProfileResult:
        """Create a result for unsupported operations."""
        if self.verbose:
            print(f"  Skipping {step_name}: {reason}")

        return ProfileResult(
            step_name=step_name,
            operation=operation,
            mean_time=0.0,
            std_time=0.0,
            min_time=0.0,
            max_time=0.0,
            num_runs=0,
            step_type=step_type,
        )

    def _create_failed_result(
        self, step_name: str, operation: str, step_type: str, reason: str
    ) -> ProfileResult:
        """Create a result for failed operations."""
        if self.verbose:
            print(f"  Failed {step_name}: {reason}")

        return ProfileResult(
            step_name=step_name,
            operation=operation,
            mean_time=float("nan"),  # Use NaN instead of inf for failed operations
            std_time=float("nan"),
            min_time=float("nan"),
            max_time=float("nan"),
            num_runs=0,
            step_type=step_type,
        )

    def _profile_pipeline_steps(
        self,
        pipeline: Pipeline,
        operation: str,
        X: Any,
        y: Any | None = None,
        is_fitted: bool = False,
    ) -> list[ProfileResult]:
        """Profile individual steps in a Pipeline."""
        children = []

        # For fitted pipelines doing predict/transform operations,
        # we need to process data through the pipeline sequentially
        if is_fitted and operation in ["predict", "transform"]:
            current_X = X

            for i, (step_name, step) in enumerate(pipeline.steps):
                try:
                    # For intermediate steps, time the transform operation
                    if i < len(pipeline.steps) - 1:
                        if hasattr(step, "transform"):
                            # Create a closure that captures current_X
                            def make_transform_func(
                                step_obj: Any, data: Any
                            ) -> Callable[[], Any]:
                                return lambda: step_obj.transform(data)

                            # Time the transform operation
                            times = self._time_operation(
                                make_transform_func(step, current_X)
                            )
                            if times:
                                times_array = np.array(times)
                                child_result = ProfileResult(
                                    step_name=step_name,
                                    operation="transform",
                                    mean_time=float(np.mean(times_array)),
                                    std_time=float(np.std(times_array)),
                                    min_time=float(np.min(times_array)),
                                    max_time=float(np.max(times_array)),
                                    num_runs=len(times),
                                    step_type=self._get_step_type(step),
                                )
                                # Update current_X for next step (do this outside timing)
                                current_X = step.transform(current_X)
                            else:
                                child_result = self._create_failed_result(
                                    step_name,
                                    "transform",
                                    self._get_step_type(step),
                                    "Transform timing failed",
                                )
                        else:
                            child_result = self._create_unsupported_result(
                                step_name,
                                "transform",
                                self._get_step_type(step),
                                "No transform method",
                            )
                    else:
                        # Final step - use the requested operation
                        if operation == "predict" and hasattr(step, "predict"):

                            def make_predict_func(
                                step_obj: Any, data: Any
                            ) -> Callable[[], Any]:
                                return lambda: step_obj.predict(data)

                            times = self._time_operation(
                                make_predict_func(step, current_X)
                            )
                        elif operation == "transform" and hasattr(step, "transform"):

                            def make_transform_func(
                                step_obj: Any, data: Any
                            ) -> Callable[[], Any]:
                                return lambda: step_obj.transform(data)

                            times = self._time_operation(
                                make_transform_func(step, current_X)
                            )
                        else:
                            child_result = self._create_unsupported_result(
                                step_name,
                                operation,
                                self._get_step_type(step),
                                f"No {operation} method",
                            )
                            children.append(child_result)
                            continue

                        if times:
                            times_array = np.array(times)
                            child_result = ProfileResult(
                                step_name=step_name,
                                operation=operation,
                                mean_time=float(np.mean(times_array)),
                                std_time=float(np.std(times_array)),
                                min_time=float(np.min(times_array)),
                                max_time=float(np.max(times_array)),
                                num_runs=len(times),
                                step_type=self._get_step_type(step),
                            )
                        else:
                            child_result = self._create_failed_result(
                                step_name,
                                operation,
                                self._get_step_type(step),
                                f"{operation} timing failed",
                            )

                    # Add children for composite steps
                    if isinstance(step, Pipeline):
                        child_result.children = self._profile_pipeline_steps(
                            step,
                            "transform" if i < len(pipeline.steps) - 1 else operation,
                            current_X,
                            y,
                            True,
                        )
                    elif isinstance(step, ColumnTransformer):
                        child_result.children = self._profile_column_transformer_steps(
                            step,
                            "transform" if i < len(pipeline.steps) - 1 else operation,
                            current_X,
                            y,
                            True,
                        )

                    children.append(child_result)

                except Exception as e:
                    if self.verbose:
                        print(f"  Error profiling step {step_name}: {e}")
                    child_result = self._create_failed_result(
                        step_name, operation, self._get_step_type(step), f"Error: {e}"
                    )
                    children.append(child_result)
                    break  # Stop processing if a step fails

        else:
            # For fit operations or unfitted pipelines, profile each step independently
            # But skip steps that can't handle raw data appropriately
            for i, (step_name, step) in enumerate(pipeline.steps):
                # Create a copy of the step to avoid side effects
                step_copy = copy.deepcopy(step)

                # For intermediate steps, we typically use transform
                # For the final step, we use the requested operation
                if i < len(pipeline.steps) - 1:
                    # Intermediate step - use transform if available, otherwise fit
                    if hasattr(step, "transform") and operation != "fit":
                        child_operation = "transform"
                    elif hasattr(step, "fit"):
                        child_operation = "fit"
                    else:
                        # Skip this step if it doesn't support needed operations
                        child_result = self._create_unsupported_result(
                            step_name,
                            "skipped",
                            self._get_step_type(step),
                            "No transform or fit method",
                        )
                        children.append(child_result)
                        continue
                else:
                    # Final step - use requested operation, but handle special cases
                    if operation == "fit" and (
                        i > 0
                        and hasattr(step, "predict")
                        and not hasattr(step, "transform")
                    ):
                        # For final step in fit mode, we need to check if it can handle raw data
                        # If this is a classifier/regressor at the end of a preprocessing pipeline,
                        # it likely expects processed data, so we'll skip individual profiling
                        # This looks like a predictor that needs preprocessed data
                        child_result = self._create_unsupported_result(
                            step_name,
                            operation,
                            self._get_step_type(step),
                            "Predictor requires preprocessed data",
                        )
                        children.append(child_result)
                        continue
                    child_operation = operation

                child_result = self._profile_step(
                    step_copy, step_name, child_operation, X, y, is_fitted
                )
                children.append(child_result)

        return children

    def _profile_column_transformer_steps(
        self,
        column_transformer: ColumnTransformer,
        operation: str,
        X: Any,
        y: Any | None = None,
        is_fitted: bool = False,
    ) -> list[ProfileResult]:
        """Profile individual transformers in a ColumnTransformer."""
        children = []

        for name, transformer, columns in column_transformer.transformers_:
            if transformer == "drop":
                continue
            elif transformer == "passthrough":
                # Create a simple passthrough result
                child_result = ProfileResult(
                    step_name=f"{name}_passthrough",
                    operation=operation,
                    mean_time=0.0,
                    std_time=0.0,
                    min_time=0.0,
                    max_time=0.0,
                    num_runs=self.profile_runs,
                    step_type="passthrough",
                )
            else:
                try:
                    # Extract columns for this transformer
                    if hasattr(X, "iloc"):  # pandas DataFrame
                        if isinstance(columns, (list, tuple)):
                            # Check if columns are names or indices
                            if all(isinstance(col, str) for col in columns):
                                X_subset = X[columns]  # Use column names
                            else:
                                X_subset = X.iloc[:, columns]  # Use indices
                        else:
                            # Single column
                            if isinstance(columns, str):
                                X_subset = X[[columns]]  # Use column name
                            else:
                                X_subset = X.iloc[:, [columns]]  # Use index
                    else:  # numpy array
                        # Handle numpy array indexing more carefully
                        if isinstance(columns, (list, tuple)):
                            # Make sure indices are within bounds
                            max_cols = X.shape[1] if len(X.shape) > 1 else 1
                            valid_columns = [
                                col
                                for col in columns
                                if isinstance(col, int) and 0 <= col < max_cols
                            ]
                            if valid_columns:
                                X_subset = (
                                    X[:, valid_columns]
                                    if len(valid_columns) > 1
                                    else X[:, [valid_columns[0]]]
                                )
                            else:
                                raise ValueError(f"Invalid column indices: {columns}")
                        else:
                            # Single column
                            if isinstance(columns, int) and 0 <= columns < X.shape[1]:
                                X_subset = X[:, [columns]]
                            else:
                                raise ValueError(f"Invalid column index: {columns}")

                    # For fitted transformers, we can profile them directly with appropriate data
                    if is_fitted and hasattr(transformer, operation):
                        # Time the operation directly on the fitted transformer
                        if operation == "transform":

                            def make_transform_func(
                                trans_obj: Any, data: Any
                            ) -> Callable[[], Any]:
                                return lambda: trans_obj.transform(data)

                            times = self._time_operation(
                                make_transform_func(transformer, X_subset)
                            )
                        elif operation == "predict":

                            def make_predict_func(
                                trans_obj: Any, data: Any
                            ) -> Callable[[], Any]:
                                return lambda: trans_obj.predict(data)

                            times = self._time_operation(
                                make_predict_func(transformer, X_subset)
                            )
                        else:
                            # For fit operations, create a copy
                            transformer_copy = copy.deepcopy(transformer)

                            def make_fit_func(
                                trans_obj: Any, data: Any, target: Any
                            ) -> Callable[[], Any]:
                                return lambda: trans_obj.fit(data, target)

                            times = self._time_operation(
                                make_fit_func(transformer_copy, X_subset, y)
                            )

                        if times:
                            times_array = np.array(times)
                            child_result = ProfileResult(
                                step_name=name,
                                operation=operation,
                                mean_time=float(np.mean(times_array)),
                                std_time=float(np.std(times_array)),
                                min_time=float(np.min(times_array)),
                                max_time=float(np.max(times_array)),
                                num_runs=len(times),
                                step_type=self._get_step_type(transformer),
                            )
                        else:
                            child_result = self._create_failed_result(
                                name,
                                operation,
                                self._get_step_type(transformer),
                                f"{operation} timing failed",
                            )
                    else:
                        # For unfitted transformers or fit operations, use the general profiling method
                        transformer_copy = copy.deepcopy(transformer)
                        child_result = self._profile_step(
                            transformer_copy, name, operation, X_subset, y, is_fitted
                        )

                    # Add children for composite transformers
                    if isinstance(transformer, Pipeline):
                        child_result.children = self._profile_pipeline_steps(
                            transformer, operation, X_subset, y, is_fitted
                        )
                    elif isinstance(transformer, ColumnTransformer):
                        child_result.children = self._profile_column_transformer_steps(
                            transformer, operation, X_subset, y, is_fitted
                        )

                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Could not profile {name}: {e}")
                    child_result = self._create_failed_result(
                        name, operation, self._get_step_type(transformer), f"Error: {e}"
                    )

            children.append(child_result)

        return children

    def profile(
        self,
        pipeline: Pipeline | ColumnTransformer | BaseEstimator,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        operations: str | list[str] = "transform",
    ) -> dict[str, ProfileResult]:
        """
        Profile a pipeline for specified operations.

        Parameters:
        -----------
        pipeline : Union[Pipeline, ColumnTransformer, BaseEstimator]
            The pipeline to profile
        X : Union[np.ndarray, pd.DataFrame]
            Input data
        y : Optional[Union[np.ndarray, pd.Series]], default=None
            Target data for supervised operations
        operations : Union[str, List[str]], default='transform'
            Operations to profile ('fit', 'transform', 'predict', 'fit_transform')

        Returns:
        --------
        Dict[str, ProfileResult]
            Dictionary mapping operation names to their profiling results
        """
        if isinstance(operations, str):
            operations = [operations]

        results = {}

        for operation in operations:
            if self.verbose:
                print(f"\nProfiling operation: {operation}")

            # Create a fresh copy for each operation
            pipeline_copy = copy.deepcopy(pipeline)
            is_fitted = False

            # For operations that require fitting, fit the pipeline first
            if operation in ["transform", "predict"] and hasattr(pipeline_copy, "fit"):
                try:
                    if self.verbose:
                        print("Fitting pipeline...")
                    pipeline_copy.fit(X, y)
                    is_fitted = True
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not fit pipeline: {e}")
                    # Continue anyway, let individual steps handle the error

            result = self._profile_step(
                step=pipeline_copy,
                step_name="root_pipeline",
                operation=operation,
                X=X,
                y=y,
                is_fitted=is_fitted,
            )

            results[operation] = result

        self.results = results
        return results

    def profile_single_row(
        self,
        pipeline: Pipeline | ColumnTransformer | BaseEstimator,
        X_samples: list[Any] | np.ndarray | pd.DataFrame,
        y_samples: list[Any] | np.ndarray | pd.Series | None = None,
        operations: str | list[str] = "transform",
    ) -> dict[str, ProfileResult]:
        """
        Profile a pipeline for single-row operations over multiple samples.

        This method is specifically designed for testing single-row performance
        with proper cache invalidation between samples.

        Parameters:
        -----------
        pipeline : Union[Pipeline, ColumnTransformer, BaseEstimator]
            The pipeline to profile
        X_samples : Union[List, np.ndarray, pd.DataFrame]
            Multiple input samples to test
        y_samples : Optional[Union[List, np.ndarray, pd.Series]], default=None
            Corresponding target samples
        operations : Union[str, List[str]], default='transform'
            Operations to profile

        Returns:
        --------
        Dict[str, ProfileResult]
            Dictionary mapping operation names to their profiling results
        """
        if isinstance(operations, str):
            operations = [operations]

        # Convert samples to list of individual samples
        if isinstance(X_samples, (np.ndarray, pd.DataFrame)):
            if len(X_samples.shape) == 1:
                X_list = [X_samples.reshape(1, -1)]
            else:
                X_list = [X_samples[i : i + 1] for i in range(len(X_samples))]
        else:
            X_list = X_samples

        if y_samples is not None:
            if isinstance(y_samples, (np.ndarray, pd.Series)):
                y_list = [y_samples[i : i + 1] for i in range(len(y_samples))]
            else:
                y_list = y_samples
        else:
            y_list = [None] * len(X_list)

        # Fit the pipeline with all data first
        pipeline_copy = copy.deepcopy(pipeline)
        if hasattr(pipeline_copy, "fit") and any(
            op in ["transform", "predict"] for op in operations
        ):
            if self.verbose:
                print("Fitting pipeline...")

            # Use all samples for fitting
            if (
                isinstance(X_samples, (np.ndarray, pd.DataFrame))
                and len(X_samples.shape) > 1
            ):
                X_fit = X_samples
                y_fit = y_samples
            else:
                X_fit = (
                    np.vstack(X_list)
                    if isinstance(X_list[0], np.ndarray)
                    else pd.concat(X_list)
                )
                y_fit = np.concatenate(y_list) if y_list[0] is not None else None

            pipeline_copy.fit(X_fit, y_fit)

        results = {}

        for operation in operations:
            if self.verbose:
                print(f"\nProfiling single-row operation: {operation}")

            all_times = []

            # Test each sample individually
            for i, (X_sample, y_sample) in enumerate(zip(X_list, y_list)):
                if self.verbose and i % max(1, len(X_list) // 10) == 0:
                    print(f"  Sample {i + 1}/{len(X_list)}")

                # Create operation function for fitted pipeline
                if operation == "fit":

                    def op_func(
                        x_sample: Any = X_sample, y_sample: Any = y_sample
                    ) -> Any:
                        temp_pipeline = copy.deepcopy(pipeline_copy)
                        return temp_pipeline.fit(x_sample, y_sample)
                elif operation == "transform":

                    def op_func(x_sample: Any = X_sample, y_sample: Any = None) -> Any:  # noqa: ARG001
                        return pipeline_copy.transform(x_sample)
                elif operation == "predict":

                    def op_func(x_sample: Any = X_sample, y_sample: Any = None) -> Any:  # noqa: ARG001
                        return pipeline_copy.predict(x_sample)
                elif operation == "fit_transform":

                    def op_func(
                        x_sample: Any = X_sample, y_sample: Any = y_sample
                    ) -> Any:
                        temp_pipeline = copy.deepcopy(pipeline_copy)
                        return temp_pipeline.fit_transform(x_sample, y_sample)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

                # Time this sample with fewer runs (since we have multiple samples)
                sample_times = self._time_operation(op_func)
                all_times.extend(sample_times)

            # Calculate overall statistics
            if all_times:
                times_array = np.array(all_times)
                result = ProfileResult(
                    step_name="single_row_pipeline",
                    operation=operation,
                    mean_time=float(np.mean(times_array)),
                    std_time=float(np.std(times_array)),
                    min_time=float(np.min(times_array)),
                    max_time=float(np.max(times_array)),
                    num_runs=len(all_times),
                    step_type=self._get_step_type(pipeline),
                )
            else:
                result = ProfileResult(
                    step_name="single_row_pipeline",
                    operation=operation,
                    mean_time=float("nan"),  # Use NaN instead of inf
                    std_time=float("nan"),
                    min_time=float("nan"),
                    max_time=float("nan"),
                    num_runs=0,
                    step_type=self._get_step_type(pipeline),
                )

            results[operation] = result

        self.results = results
        return results

    def print_report(
        self, results: dict[str, ProfileResult] | None = None, _indent: int = 0
    ) -> None:
        """
        Print a hierarchical report of profiling results.

        Parameters:
        -----------
        results : Optional[Dict[str, ProfileResult]], default=None
            Results to print. If None, uses self.results
        indent : int, default=0
            Indentation level for hierarchical display
        """
        if results is None:
            results = self.results

        for operation, result in results.items():
            print(f"\n{'=' * 60}")
            print(f"🚀 {operation.upper()} OPERATION")
            print(f"{'=' * 60}")
            self._print_result(result, 0)

    def _print_result(self, result: ProfileResult, indent: int = 0) -> None:
        """Print a single ProfileResult with hierarchy."""
        indent_str = "  " * indent

        # Handle NaN values
        if np.isnan(result.mean_time):
            prefix = "├─ " if indent > 0 else ""
            print(
                f"{indent_str}{prefix}{result.step_name} ({result.step_type}): FAILED"
            )
            return

        # Get formatted times
        formatted_times = result.get_formatted_times()

        # Print component name and main timing info
        if indent == 0:
            print(f"{indent_str}{result.step_name} ({result.step_type})")
            print(
                f"{indent_str}  ⏱️  {formatted_times['mean']} ± {formatted_times['std']} "
                f"[{formatted_times['min']} - {formatted_times['max']}] ({result.num_runs} runs)"
            )
        else:
            print(
                f"{indent_str}├─ {result.step_name} ({result.step_type}): "
                f"{formatted_times['mean']} ± {formatted_times['std']} "
                f"[{formatted_times['min']}-{formatted_times['max']}]"
            )

        if result.children:
            for _i, child in enumerate(result.children):
                # Add vertical line continuation for nested children
                if indent > 0:
                    child_indent_str = "  " * indent + "│  "
                    # Replace the base indent with the continuation line
                    old_indent = "  " * (indent + 1)
                    new_child_result = self._get_child_with_indent(
                        child, child_indent_str, old_indent
                    )
                    self._print_child_result(
                        new_child_result, indent + 1, child_indent_str
                    )
                else:
                    self._print_result(child, indent + 1)

    def _print_child_result(
        self, result: ProfileResult, indent: int, custom_indent: str
    ) -> None:
        """Print a child result with custom indentation for tree structure."""
        # Handle NaN values
        if np.isnan(result.mean_time):
            print(f"{custom_indent}├─ {result.step_name} ({result.step_type}): FAILED")
            return

        # Get formatted times
        formatted_times = result.get_formatted_times()

        print(
            f"{custom_indent}├─ {result.step_name} ({result.step_type}): "
            f"{formatted_times['mean']} ± {formatted_times['std']} "
            f"[{formatted_times['min']}-{formatted_times['max']}]"
        )

        if result.children:
            for child in result.children:
                child_indent = custom_indent + "│  "
                self._print_child_result(child, indent + 1, child_indent)

    def _get_child_with_indent(
        self, result: ProfileResult, _child_indent: str, _old_indent: str
    ) -> ProfileResult:
        """Helper to maintain the same result but with different formatting context."""
        return result  # The result itself doesn't change, just how we print it

    def to_dataframe(
        self, results: dict[str, ProfileResult] | None = None
    ) -> pd.DataFrame:
        """
        Convert profiling results to a pandas DataFrame.

        Parameters:
        -----------
        results : Optional[Dict[str, ProfileResult]], default=None
            Results to convert. If None, uses self.results

        Returns:
        --------
        pd.DataFrame
            DataFrame with profiling results
        """
        if results is None:
            results = self.results

        rows: list[dict[str, Any]] = []

        for operation, result in results.items():
            self._add_result_to_rows(result, rows, operation, "")

        return pd.DataFrame(rows)

    def _add_result_to_rows(
        self,
        result: ProfileResult,
        rows: list[dict[str, Any]],
        operation: str,
        parent_path: str,
    ) -> None:
        """Recursively add results to rows list."""
        current_path = (
            f"{parent_path}.{result.step_name}" if parent_path else result.step_name
        )

        # Get formatted times
        formatted_times = result.get_formatted_times()

        rows.append(
            {
                "operation": operation,
                "path": current_path,
                "step_name": result.step_name,
                "step_type": result.step_type,
                "mean_time": result.mean_time,
                "std_time": result.std_time,
                "min_time": result.min_time,
                "max_time": result.max_time,
                "mean_time_formatted": formatted_times["mean"],
                "std_time_formatted": formatted_times["std"],
                "min_time_formatted": formatted_times["min"],
                "max_time_formatted": formatted_times["max"],
                "num_runs": result.num_runs,
            }
        )

        for child in result.children:
            self._add_result_to_rows(child, rows, operation, current_path)


@dataclass
class CompiledProfileResult(ProfileResult):
    """Extended ProfileResult for compiled pipeline comparisons."""

    original_time: float = 0.0
    compiled_time: float = 0.0
    speedup: float = 1.0
    original_std: float = 0.0
    compiled_std: float = 0.0

    def get_speedup_info(self) -> dict[str, str]:
        """Get formatted speedup information."""
        if self.compiled_time > 0:
            speedup = self.original_time / self.compiled_time
        else:
            speedup = float("inf")

        return {
            "speedup": f"{speedup:.2f}x",
            "improvement": f"{((speedup - 1) * 100):.1f}%"
            if speedup > 1
            else f"{((1 - speedup) * 100):.1f}% slower",
        }


class CompiledPipelineProfiler:
    """
    Profiler specifically designed for compiled pipelines with comparison capabilities.

    This profiler extends the existing PipelineProfiler to provide clean profiling
    of compiled pipelines without modifying the core compilation logic.
    """

    def __init__(
        self,
        warmup_runs: int = 5,
        profile_runs: int = 50,
        cache_invalidation: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the compiled pipeline profiler.

        Parameters:
        -----------
        warmup_runs : int, default=5
            Number of warmup runs (more for compiled functions)
        profile_runs : int, default=50
            Number of runs for timing (more for better statistics on fast functions)
        cache_invalidation : bool, default=True
            Whether to perform cache invalidation between runs
        verbose : bool, default=False
            Whether to print progress information
        """
        self.warmup_runs = warmup_runs
        self.profile_runs = profile_runs
        self.cache_invalidation = cache_invalidation
        self.verbose = verbose

        # Create base profiler for original pipeline profiling
        self.base_profiler = PipelineProfiler(
            warmup_runs=warmup_runs,
            profile_runs=profile_runs,
            cache_invalidation=cache_invalidation,
            verbose=verbose,
        )

    def _invalidate_caches(self) -> None:
        """Invalidate CPU caches and trigger garbage collection."""
        if self.cache_invalidation:
            gc.collect()
            # Smaller dummy operations for compiled function profiling
            dummy_array = np.random.random((100, 10))
            _ = np.sum(dummy_array)
            del dummy_array
            gc.collect()

    def _time_compiled_operation(
        self, compiled_fn: Callable[..., Any], data_samples: list[Any]
    ) -> list[float]:
        """
        Time a compiled function across multiple data samples.

        Parameters:
        -----------
        compiled_fn : Callable
            The compiled function to time
        data_samples : List[Any]
            List of input samples to test with

        Returns:
        --------
        List[float]
            List of timing measurements
        """
        times = []

        # Warmup runs
        for _ in range(self.warmup_runs):
            for sample in data_samples[
                : min(5, len(data_samples))
            ]:  # Use first few samples for warmup
                try:
                    compiled_fn(sample)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning during warmup: {e}")
            self._invalidate_caches()

        # Actual timing runs
        for i in range(self.profile_runs):
            if self.verbose and i % max(1, self.profile_runs // 10) == 0:
                print(f"  Compiled function run {i + 1}/{self.profile_runs}")

            # Use a random sample from the data to avoid cache effects
            sample = data_samples[i % len(data_samples)]

            self._invalidate_caches()

            start_time = time.perf_counter()
            try:
                compiled_fn(sample)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # Sanity check for timing (compiled functions should be very fast)
                if elapsed_time >= 0 and elapsed_time < 1.0:  # Less than 1 second
                    times.append(elapsed_time)
                elif self.verbose:
                    print(f"  Warning: Suspicious timing {elapsed_time:.6f}s, skipping")

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Compiled function failed: {e}")

        return times

    def _prepare_data_samples(
        self, X: np.ndarray | pd.DataFrame | list[Any], num_samples: int = 20
    ) -> list[Any]:
        """
        Prepare individual data samples for single-row profiling.

        Parameters:
        -----------
        X : Union[np.ndarray, pd.DataFrame, List]
            Input data
        num_samples : int, default=20
            Number of samples to prepare

        Returns:
        --------
        List[Any]
            List of individual samples prepared for single-row input
        """
        if isinstance(X, pd.DataFrame):
            # Convert DataFrame rows to lists or dicts
            samples = []
            for i in range(min(num_samples, len(X))):
                row = X.iloc[i]
                # You can choose to use dict or list format
                samples.append(row.tolist())  # Use list format
                # samples.append(row.to_dict())  # Alternative: dict format
            return samples

        elif isinstance(X, np.ndarray):
            # Convert numpy array rows to lists
            if len(X.shape) == 1:
                return [X.tolist()]
            else:
                samples = []
                for i in range(min(num_samples, len(X))):
                    samples.append(X[i].tolist())
                return samples

        elif isinstance(X, list):
            # Already in list format
            return X[:num_samples]

        else:
            raise ValueError(f"Unsupported data type: {type(X)}")

    def profile_compiled_vs_original(
        self,
        pipeline: Pipeline | ColumnTransformer,
        X: np.ndarray | pd.DataFrame | list[Any],
        y: np.ndarray | pd.Series | list[Any] | None = None,
        operation: str = "predict",
        num_samples: int = 20,
    ) -> CompiledProfileResult:
        """
        Profile a compiled pipeline against its original version.

        Parameters:
        -----------
        pipeline : Union[Pipeline, ColumnTransformer]
            The fitted pipeline to profile
        X : Union[np.ndarray, pd.DataFrame, List]
            Input data
        y : Optional[Union[np.ndarray, pd.Series]], default=None
            Target data (if needed)
        operation : str, default='predict'
            Operation to profile ('predict' or 'transform')
        num_samples : int, default=20
            Number of data samples to use for testing

        Returns:
        --------
        CompiledProfileResult
            Profiling results comparing original vs compiled performance
        """
        if self.verbose:
            print(f"Profiling compiled pipeline vs original for {operation} operation")

        # Prepare data samples for single-row testing
        data_samples = self._prepare_data_samples(X, num_samples)

        # Profile original pipeline using single-row profiling
        if self.verbose:
            print("  Profiling original pipeline...")

        # Convert data_samples back to appropriate format for base profiler
        if isinstance(X, pd.DataFrame):
            # Reconstruct DataFrame samples for base profiler
            X_samples_df = []
            for sample in data_samples:
                if isinstance(sample, list):
                    # Convert list back to DataFrame row
                    sample_df = pd.DataFrame([sample], columns=X.columns)
                    X_samples_df.append(sample_df)
                else:
                    X_samples_df.append(sample)
            original_samples = X_samples_df
        elif isinstance(X, np.ndarray):
            # Convert list samples back to numpy arrays
            original_samples = [
                np.array(sample).reshape(1, -1) for sample in data_samples
            ]
        else:
            original_samples = data_samples

        # Prepare corresponding y samples if provided
        y_samples = None
        if y is not None:
            if isinstance(y, (np.ndarray, pd.Series)):
                y_samples = [y[i : i + 1] for i in range(min(num_samples, len(y)))]
            else:  # For list or other sequence types
                y_samples = y[:num_samples]

        original_results = self.base_profiler.profile_single_row(
            pipeline=pipeline,
            X_samples=original_samples,
            y_samples=y_samples,
            operations=[operation],
        )

        original_result = original_results[operation]

        # Compile the pipeline
        if self.verbose:
            print("  Compiling pipeline...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress compilation warnings
            compiled_fn = compile_pipeline(pipeline)

        # Profile compiled function
        if self.verbose:
            print("  Profiling compiled function...")

        compiled_times = self._time_compiled_operation(compiled_fn, data_samples)

        # Calculate compiled statistics
        if compiled_times:
            compiled_times_array = np.array(compiled_times)
            compiled_mean = float(np.mean(compiled_times_array))
            compiled_std = float(np.std(compiled_times_array))
            compiled_min = float(np.min(compiled_times_array))
            compiled_max = float(np.max(compiled_times_array))
            compiled_runs = len(compiled_times)
        else:
            compiled_mean = compiled_std = compiled_min = compiled_max = float("nan")
            compiled_runs = 0

        # Calculate speedup
        if compiled_mean > 0 and not np.isnan(original_result.mean_time):
            speedup = original_result.mean_time / compiled_mean
        else:
            speedup = float("nan")

        # Create combined result
        result = CompiledProfileResult(
            step_name="compiled_pipeline",
            operation=operation,
            mean_time=compiled_mean,
            std_time=compiled_std,
            min_time=compiled_min,
            max_time=compiled_max,
            num_runs=compiled_runs,
            step_type="CompiledPipeline",
            original_time=original_result.mean_time,
            compiled_time=compiled_mean,
            speedup=speedup,
            original_std=original_result.std_time,
            compiled_std=compiled_std,
        )

        return result

    def profile_compiled_only(
        self,
        compiled_fn: Callable[..., Any],
        data_samples: list[Any],
        operation: str = "predict",
    ) -> ProfileResult:
        """
        Profile only a compiled function (when you don't need comparison).

        Parameters:
        -----------
        compiled_fn : Callable
            The compiled function to profile
        data_samples : List[Any]
            List of input samples to test with
        operation : str, default='predict'
            Operation name for reporting

        Returns:
        --------
        ProfileResult
            Profiling results for the compiled function
        """
        if self.verbose:
            print(f"Profiling compiled function for {operation} operation")

        compiled_times = self._time_compiled_operation(compiled_fn, data_samples)

        if compiled_times:
            compiled_times_array = np.array(compiled_times)
            result = ProfileResult(
                step_name="compiled_function",
                operation=operation,
                mean_time=float(np.mean(compiled_times_array)),
                std_time=float(np.std(compiled_times_array)),
                min_time=float(np.min(compiled_times_array)),
                max_time=float(np.max(compiled_times_array)),
                num_runs=len(compiled_times),
                step_type="CompiledFunction",
            )
        else:
            result = ProfileResult(
                step_name="compiled_function",
                operation=operation,
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                num_runs=0,
                step_type="CompiledFunction",
            )

        return result

    def print_comparison_report(self, result: CompiledProfileResult) -> None:
        """
        Print a detailed comparison report for compiled vs original pipeline.

        Parameters:
        -----------
        result : CompiledProfileResult
            The comparison result to report
        """
        print(f"\n{'=' * 60}")
        print(f"🚀 COMPILED PIPELINE COMPARISON - {result.operation.upper()}")
        print(f"{'=' * 60}")

        # Handle NaN/failed cases
        if np.isnan(result.original_time) or np.isnan(result.compiled_time):
            print("❌ Profiling failed - insufficient data")
            return

        # Get formatted times
        orig_formatted = result._format_time(result.original_time)
        comp_formatted = result._format_time(result.compiled_time)
        speedup_info = result.get_speedup_info()

        print("📊 PERFORMANCE COMPARISON:")
        print(
            f"  Original Pipeline:  {orig_formatted[0]:.3f}{orig_formatted[1]} ± {result._format_time(result.original_std)[0]:.3f}{result._format_time(result.original_std)[1]}"
        )
        print(
            f"  Compiled Pipeline:  {comp_formatted[0]:.3f}{comp_formatted[1]} ± {result._format_time(result.compiled_std)[0]:.3f}{result._format_time(result.compiled_std)[1]}"
        )
        print(
            f"  ⚡ Speedup:         {speedup_info['speedup']} ({speedup_info['improvement']})"
        )
        print(
            f"  📈 Runs:           {result.num_runs} compiled vs {result.num_runs} original"
        )

        # Performance category
        if result.speedup >= 10:
            emoji = "🔥"
            category = "EXCELLENT"
        elif result.speedup >= 5:
            emoji = "⚡"
            category = "VERY GOOD"
        elif result.speedup >= 2:
            emoji = "✅"
            category = "GOOD"
        elif result.speedup >= 1.1:
            emoji = "👍"
            category = "MODEST"
        else:
            emoji = "⚠️"
            category = "MINIMAL"

        print(f"  {emoji} Performance: {category}")


def profile_pipeline_compilation(
    pipeline: Pipeline | ColumnTransformer,
    X: np.ndarray | pd.DataFrame | list[Any],
    y: np.ndarray | pd.Series | None = None,
    operation: str = "predict",
    num_samples: int = 20,
    verbose: bool = True,
) -> CompiledProfileResult:
    """
    Convenience function for quick compiled pipeline profiling.

    Parameters:
    -----------
    pipeline : Union[Pipeline, ColumnTransformer]
        The fitted pipeline to profile
    X : Union[np.ndarray, pd.DataFrame, List]
        Input data
    y : Optional[Union[np.ndarray, pd.Series]], default=None
        Target data (if needed)
    operation : str, default='predict'
        Operation to profile ('predict' or 'transform')
    num_samples : int, default=20
        Number of data samples to use for testing
    verbose : bool, default=True
        Whether to print detailed output

    Returns:
    --------
    CompiledProfileResult
        Profiling results comparing original vs compiled performance
    """
    profiler = CompiledPipelineProfiler(verbose=verbose)
    result = profiler.profile_compiled_vs_original(
        pipeline=pipeline, X=X, y=y, operation=operation, num_samples=num_samples
    )

    if verbose:
        profiler.print_comparison_report(result)

    return result
