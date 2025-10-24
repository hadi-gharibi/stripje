"""Profiling utilities for scikit-learn pipelines and Stripje compiled pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter_ns
from types import MethodType
from typing import Any, Callable

import numpy as np
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .registry import create_fallback_handler, get_handler

__all__ = ["CallEvent", "ProfileNode", "ProfileReport", "PipelineProfiler"]


def _format_duration(seconds: float) -> str:
    """Return a human-readable representation of elapsed time."""
    if seconds < 0:
        return f"{seconds:.3f} s"
    if seconds < 1:
        return f"{seconds * 1_000:.3f} ms"
    return f"{seconds:.3f} s"


@dataclass
class CallEvent:
    """Timing information for a single call of a profiled node."""

    start_ns: int
    end_ns: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000_000


@dataclass
class ProfileNode:
    """Node in the profiling tree representing a pipeline component."""

    name: str
    kind: str
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent: ProfileNode | None = field(default=None, repr=False)
    children: list[ProfileNode] = field(default_factory=list)
    events: list[CallEvent] = field(default_factory=list)

    def child(
        self,
        name: str,
        kind: str,
        method: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProfileNode:
        """Get or create a named child node."""
        for existing in self.children:
            if existing.name == name:
                if metadata:
                    existing.metadata.update(metadata)
                existing.kind = kind
                existing.method = method
                return existing
        child = ProfileNode(
            name=name, kind=kind, method=method, metadata=metadata or {}
        )
        child.parent = self
        self.children.append(child)
        return child

    def add_event(self, start_ns: int, end_ns: int) -> None:
        self.events.append(CallEvent(start_ns=start_ns, end_ns=end_ns))

    @property
    def last_duration(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].duration_seconds

    @property
    def last_duration_display(self) -> str:
        return _format_duration(self.last_duration)

    @property
    def mean_duration(self) -> float:
        if not self.events:
            return 0.0
        return sum(event.duration_seconds for event in self.events) / len(self.events)

    @property
    def mean_duration_display(self) -> str:
        return _format_duration(self.mean_duration)

    @property
    def call_count(self) -> int:
        return len(self.events)


@dataclass
class ProfileReport:
    """Structured profiling results."""

    root: ProfileNode
    output: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the profiling tree."""

        def build(node: ProfileNode) -> dict[str, Any]:
            return {
                "name": node.name,
                "kind": node.kind,
                "method": node.method,
                "metadata": node.metadata,
                "call_count": node.call_count,
                "mean_duration": node.mean_duration,
                "mean_duration_display": node.mean_duration_display,
                "last_duration": node.last_duration,
                "last_duration_display": node.last_duration_display,
                "children": [build(child) for child in node.children],
            }

        return build(self.root)


class MethodTimer:
    """Context manager that patches an object's method to collect timing events."""

    def __init__(self, obj: Any, method_name: str, node: ProfileNode) -> None:
        self.obj = obj
        self.method_name = method_name
        self.node = node
        self._original_attr: Any = None
        self._owned = False
        self._bound_original: Callable[..., Any] | None = None

    def __enter__(self) -> MethodTimer:
        if not hasattr(self.obj, self.method_name):
            return self

        self._owned = self.method_name in vars(self.obj)
        if self._owned:
            self._original_attr = vars(self.obj)[self.method_name]
        self._bound_original = getattr(self.obj, self.method_name)

        if self._bound_original is not None:

            @wraps(self._bound_original)
            def wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
                start = perf_counter_ns()
                try:
                    return self._bound_original(*args, **kwargs)  # type: ignore[misc]
                finally:
                    end = perf_counter_ns()
                    self.node.add_event(start, end)

            bound_wrapper = MethodType(wrapper, self.obj)
            setattr(self.obj, self.method_name, bound_wrapper)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not hasattr(self.obj, self.method_name):
            return
        if self._owned:
            setattr(self.obj, self.method_name, self._original_attr)
        else:
            with suppress(AttributeError):
                delattr(self.obj, self.method_name)


class PipelineProfiler:
    """Profile scikit-learn pipelines and their compiled Stripje counterparts."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        mode: str = "transform",
        repetitions: int = 1,
        warmup: int = 0,
    ) -> None:
        self.pipeline = pipeline
        self.mode = mode
        self.repetitions = max(1, repetitions)
        self.warmup = max(0, warmup)

    def run(self, X: Any, y: Any = None) -> ProfileReport:
        """Profile the provided pipeline on the given batch input."""

        for _ in range(self.warmup):
            self._run_steps(self.pipeline.steps, X, None, self.mode, y)

        root = ProfileNode(
            name="pipeline", kind=type(self.pipeline).__name__, method=self.mode
        )
        output: Any = None
        for _ in range(self.repetitions):
            start = perf_counter_ns()
            output = self._run_steps(self.pipeline.steps, X, root, self.mode, y)
            end = perf_counter_ns()
            root.add_event(start, end)

        return ProfileReport(root=root, output=output)

    def run_compiled(self, sample: Any) -> ProfileReport:
        """Profile the compiled single-row pipeline produced by Stripje."""

        root = ProfileNode(name="compiled_pipeline", kind="callable", method="call")
        current = sample  # Keep original format for column transformers
        start_root = perf_counter_ns()

        for name, step in self.pipeline.steps:
            current = self._profile_compiled_step(step, name, current, root)

        end_root = perf_counter_ns()
        root.add_event(start_root, end_root)
        return ProfileReport(root=root, output=current)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_compiled_step(
        self, step: Any, name: str, data: Any, parent: ProfileNode
    ) -> Any:
        """Profile a compiled step, recursing into nested structures."""
        node = parent.child(
            name=name, kind=type(step).__name__, method="call"
        )

        start = perf_counter_ns()

        # Handle nested Pipeline
        if isinstance(step, Pipeline):
            current = data
            for sub_name, sub_step in step.steps:
                current = self._profile_compiled_step(sub_step, sub_name, current, node)
            result = current

        # Handle ColumnTransformer
        elif isinstance(step, ColumnTransformer):
            result = self._profile_compiled_column_transformer(step, data, node)

        # Handle regular transformers/estimators
        else:
            handler = get_handler(type(step))
            if handler is None:
                handler = create_fallback_handler
            compiled_fn = handler(step)
            result = compiled_fn(data)
            result = self._coerce_single_sample(result)

        end = perf_counter_ns()
        node.add_event(start, end)

        return result

    def _extract_columns(self, data: Any, columns: Any) -> Any:
        """Extract specific columns from data (handles Series, DataFrame, ndarray)."""
        # Handle pandas Series (single row from DataFrame)
        if hasattr(data, 'index') and not hasattr(data, 'iloc'):
            # This is a pandas Series
            return data[columns]

        # Handle pandas DataFrame
        if hasattr(data, 'iloc'):
            return data[columns]

        # Handle numpy array - columns should be indices
        if isinstance(data, np.ndarray):
            if isinstance(columns, (list, np.ndarray)):
                return data[columns] if data.ndim == 1 else data[:, columns]
            else:
                return data[columns] if data.ndim == 1 else data[:, columns]

        return data

    def _prepare_data_for_column_transformer(self, data: Any) -> Any:
        """Convert data to appropriate format for ColumnTransformer profiling.

        Converts pandas Series to single-row DataFrame to ensure consistent
        column-based access patterns.
        """
        is_series = hasattr(data, 'index') and not hasattr(data, 'iloc')
        if is_series:
            # Convert Series to single-row DataFrame for ColumnTransformer
            import pandas as pd
            return pd.DataFrame([data])
        return data

    def _extract_and_prepare_column_data(self, data_df: Any, columns: Any) -> Any:
        """Extract columns from data and prepare single-row format if needed."""
        col_data = self._extract_columns(data_df, columns)
        # Extract single row if we have a DataFrame with one row
        if hasattr(col_data, 'iloc') and len(col_data) == 1:
            col_data = col_data.iloc[0]
        return col_data

    def _profile_drop_transformer(
        self, name: str, columns: Any, parent: ProfileNode
    ) -> None:
        """Profile a 'drop' transformer (no-op)."""
        metadata = {"columns": columns}
        child = parent.child(name=name, kind="drop", method="skip", metadata=metadata)
        noop = perf_counter_ns()
        child.add_event(noop, noop)

    def _profile_passthrough_transformer(
        self, name: str, columns: Any, data_df: Any, parent: ProfileNode
    ) -> Any:
        """Profile a 'passthrough' transformer."""
        metadata = {"columns": columns}
        child = parent.child(
            name=name, kind="passthrough", method="identity", metadata=metadata
        )
        start = perf_counter_ns()
        col_data = self._extract_and_prepare_column_data(data_df, columns)
        result = self._coerce_single_sample(col_data)
        end = perf_counter_ns()
        child.add_event(start, end)
        return result

    def _profile_pipeline_transformer(
        self, name: str, trans: Pipeline, columns: Any, data_df: Any, parent: ProfileNode
    ) -> Any:
        """Profile a nested Pipeline transformer."""
        metadata = {"columns": columns}
        child = parent.child(
            name=name, kind=type(trans).__name__, method="call", metadata=metadata
        )
        start = perf_counter_ns()

        # Extract columns and process through pipeline steps
        col_data = self._extract_and_prepare_column_data(data_df, columns)
        current = col_data
        for sub_name, sub_step in trans.steps:
            current = self._profile_compiled_step(sub_step, sub_name, current, child)

        result = self._coerce_single_sample(current)
        end = perf_counter_ns()
        child.add_event(start, end)
        return result

    def _profile_regular_transformer(
        self, name: str, trans: Any, columns: Any, data_df: Any, parent: ProfileNode
    ) -> Any:
        """Profile a regular (non-Pipeline) transformer."""
        metadata = {"columns": columns}
        child = parent.child(
            name=name, kind=type(trans).__name__, method="call", metadata=metadata
        )
        start = perf_counter_ns()

        # Extract columns and apply transformer
        col_data = self._extract_and_prepare_column_data(data_df, columns)
        handler = get_handler(type(trans))
        if handler is None:
            handler = create_fallback_handler
        compiled_fn = handler(trans)
        result = compiled_fn(self._coerce_single_sample(col_data))
        result = self._coerce_single_sample(result)

        end = perf_counter_ns()
        child.add_event(start, end)
        return result

    def _combine_transformer_results(self, results: list[Any], data: Any) -> Any:
        """Combine results from multiple transformers into a single array."""
        if results:
            combined = np.concatenate([r.ravel() if r.ndim > 1 else r for r in results])
            return combined
        return data

    def _profile_compiled_column_transformer(
        self, transformer: ColumnTransformer, data: Any, parent: ProfileNode
    ) -> Any:
        """Profile compiled ColumnTransformer with its sub-transformers."""
        # Prepare data format for consistent column access
        data_df = self._prepare_data_for_column_transformer(data)
        results = []

        for name, trans, columns in transformer.transformers_:
            if trans == "drop":
                self._profile_drop_transformer(name, columns, parent)
                continue

            if trans == "passthrough":
                result = self._profile_passthrough_transformer(name, columns, data_df, parent)
                results.append(result)
                continue

            # Handle sub-pipeline or regular transformer
            if isinstance(trans, Pipeline):
                result = self._profile_pipeline_transformer(name, trans, columns, data_df, parent)
                results.append(result)
            else:
                result = self._profile_regular_transformer(name, trans, columns, data_df, parent)
                results.append(result)

        return self._combine_transformer_results(results, data)

    def _compiled_steps(self) -> list[tuple[str, Callable[[Any], Any]]]:
        compiled_steps: list[tuple[str, Callable[[Any], Any]]] = []
        for name, step in self.pipeline.steps:
            handler = get_handler(type(step))
            if handler is None:
                handler = create_fallback_handler
            compiled_steps.append((name, handler(step)))
        return compiled_steps

    def _run_steps(
        self,
        steps: Iterable[tuple[str, Any]],
        data: Any,
        parent: ProfileNode | None,
        final_method: str,
        y: Any,
    ) -> Any:
        current = data
        parent_node = parent or ProfileNode(
            name="pipeline", kind="Pipeline", method=final_method
        )

        steps_list = list(steps)
        for idx, (name, step) in enumerate(steps_list):
            is_last = idx == len(steps_list) - 1
            method = self._determine_method(step, is_last, final_method)
            metadata = {"method": method}
            node = parent_node.child(
                name=name, kind=type(step).__name__, method=method, metadata=metadata
            )

            start = perf_counter_ns()
            if isinstance(step, Pipeline):
                current = self._run_steps(step.steps, current, node, method, y)
            elif isinstance(step, ColumnTransformer):
                current = self._execute_column_transformer(step, current, node, method)
            else:
                call_args = (current,) if method != "fit" else (current, y)
                attr = getattr(step, method)
                current = attr(*call_args)
            end = perf_counter_ns()
            node.add_event(start, end)

        return current

    def _execute_column_transformer(
        self,
        transformer: ColumnTransformer,
        data: Any,
        node: ProfileNode,
        method: str,
    ) -> Any:
        if method not in {"transform", "fit_transform"}:
            attr = getattr(transformer, method)
            return attr(data)

        with ExitStack() as stack:
            for name, trans, columns in transformer.transformers_:
                metadata = {"columns": columns}
                if trans == "drop":
                    child = node.child(
                        name=name, kind="drop", method="skip", metadata=metadata
                    )
                    if not child.events:
                        noop = perf_counter_ns()
                        child.add_event(noop, noop)
                    continue

                if trans == "passthrough":
                    child = node.child(
                        name=name,
                        kind="passthrough",
                        method="identity",
                        metadata=metadata,
                    )
                    if not child.events:
                        noop = perf_counter_ns()
                        child.add_event(noop, noop)
                    continue

                child_method = method if hasattr(trans, method) else "transform"
                child = node.child(
                    name=name,
                    kind=type(trans).__name__,
                    method=child_method,
                    metadata=metadata,
                )

                if hasattr(trans, "steps"):
                    stack.enter_context(MethodTimer(trans, child_method, child))
                    self._instrument_pipeline_steps(trans, child, child_method, stack)
                else:
                    stack.enter_context(MethodTimer(trans, child_method, child))

            attr = getattr(transformer, method)
            with parallel_backend("threading"):
                result = attr(data)

        for child in node.children:
            if child.parent is node and not child.events:
                noop = perf_counter_ns()
                child.add_event(noop, noop)

        return result

    def _determine_method(self, step: Any, is_last: bool, final_method: str) -> str:
        if isinstance(step, Pipeline):
            return final_method

        if is_last:
            if final_method == "predict" and hasattr(step, "predict"):
                return "predict"
            if final_method == "transform" and hasattr(step, "transform"):
                return "transform"
            if hasattr(step, final_method):
                return final_method

        if hasattr(step, "transform"):
            return "transform"
        if hasattr(step, final_method):
            return final_method
        if hasattr(step, "predict"):
            return "predict"

        raise AttributeError(
            f"Cannot determine method for step {step!r}; expected one of transform/predict/fit."
        )

    def _coerce_single_sample(self, sample: Any) -> Any:
        """Return a single sample in a numpy-friendly format for compiled steps."""
        if isinstance(sample, np.ndarray):
            return sample
        if hasattr(sample, "to_numpy"):
            arr = sample.to_numpy()
            return arr
        if isinstance(sample, dict):
            return np.asarray(list(sample.values()))
        return np.asarray(sample)

    def _instrument_pipeline_steps(
        self,
        pipeline: Any,
        parent_node: ProfileNode,
        final_method: str,
        stack: ExitStack,
    ) -> None:
        if not hasattr(pipeline, "steps"):
            return

        steps_list = list(pipeline.steps)
        for idx, (name, step) in enumerate(steps_list):
            method = self._determine_method(
                step, idx == len(steps_list) - 1, final_method
            )
            metadata = {"method": method}
            step_node = parent_node.child(
                name=name,
                kind=type(step).__name__,
                method=method,
                metadata=metadata,
            )

            if isinstance(step, Pipeline):
                stack.enter_context(MethodTimer(step, method, step_node))
                self._instrument_pipeline_steps(step, step_node, method, stack)
            elif isinstance(step, ColumnTransformer):
                stack.enter_context(MethodTimer(step, method, step_node))
            else:
                stack.enter_context(MethodTimer(step, method, step_node))
