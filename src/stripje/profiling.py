"""Profiling utilities for scikit-learn pipelines and Stripje compiled pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter_ns
from types import MethodType
from typing import Any, Callable

import numpy as np
from joblib import parallel_backend # type: ignore[import-untyped]
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

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        import uuid
        from pathlib import Path

        container_id = f"stripje-container-{uuid.uuid4().hex[:8]}"

        # Load CSS from external file
        css_path = Path(__file__).parent / "profiling_style.css"
        try:
            with css_path.open() as f:
                style = f.read()
        except FileNotFoundError:
            style = "/* CSS file not found */"

        # Replace placeholder class in CSS
        style = style.replace(".stripje-profile-container", f".{container_id}")

        html_parts = [f"<style>{style}</style>"]
        html_parts.append(
            f'<div class="{container_id} stripje-profile-container sk-top-container">'
        )

        # Fallback text (hidden by CSS)
        html_parts.append('<div class="sk-text-repr-fallback">')
        html_parts.append(
            f"<pre>Pipeline Profile: {self._escape_html(self.root.name)}</pre>"
        )
        html_parts.append("</div>")

        # Main container
        estimator_id_counter = {"count": 0}
        html_parts.append('<div class="sk-container">')
        html_parts.append(
            self._render_pipeline_wrapper(self.root, estimator_id_counter)
        )
        html_parts.append("</div>")
        html_parts.append("</div>")

        return "\n".join(html_parts)

    def _render_pipeline_wrapper(
        self, node: ProfileNode, estimator_id_counter: dict[str, int]
    ) -> str:
        """Render the main pipeline wrapper with dashed border."""
        html = ['<div class="sk-item sk-dashed-wrapped">']

        # Pipeline label
        html.append('<div class="sk-label-container">')
        timing_badge = self._get_timing_badge(
            node.mean_duration, node.mean_duration_display
        )
        html.append('<div class="sk-label">')
        html.append(
            f'<span class="sk-label-text">{self._escape_html(node.name)}</span>'
        )
        html.append(timing_badge)
        html.append("</div>")
        html.append("</div>")

        # Pipeline steps
        if node.children:
            html.append('<div class="sk-serial">')
            for child in node.children:
                html.append(self._render_node(child, estimator_id_counter))
            html.append("</div>")

        html.append("</div>")
        return "\n".join(html)

    def _render_node(
        self, node: ProfileNode, estimator_id_counter: dict[str, int]
    ) -> str:
        """Render a single node (estimator or nested structure)."""

        # Check if this is a ColumnTransformer (parallel layout)
        if node.kind == "ColumnTransformer":
            return self._render_column_transformer(node, estimator_id_counter)

        # Check if this is a nested Pipeline
        if node.kind == "Pipeline" and node.children:
            return self._render_nested_pipeline(node, estimator_id_counter)

        # Regular estimator
        return self._render_estimator(node, estimator_id_counter)

    def _render_column_transformer(
        self, node: ProfileNode, estimator_id_counter: dict[str, int]
    ) -> str:
        """Render ColumnTransformer with parallel layout."""
        html = ['<div class="sk-item">']

        # Label
        html.append('<div class="sk-label-container">')
        timing_badge = self._get_timing_badge(
            node.mean_duration, node.mean_duration_display
        )
        html.append('<div class="sk-label">')
        html.append(
            f'<span class="sk-label-text">{self._escape_html(node.name)}</span>'
        )
        html.append(timing_badge)
        html.append("</div>")
        html.append("</div>")

        # Parallel items
        if node.children:
            html.append('<div class="sk-parallel">')
            for child in node.children:
                html.append('<div class="sk-parallel-item">')
                html.append(self._render_parallel_branch(child, estimator_id_counter))
                html.append("</div>")
            html.append("</div>")

        html.append("</div>")
        return "\n".join(html)

    def _render_parallel_branch(
        self, node: ProfileNode, estimator_id_counter: dict[str, int]
    ) -> str:
        """Render a single branch in a parallel layout (e.g., one transformer in ColumnTransformer)."""
        # Check if this branch itself is a Pipeline
        if node.kind == "Pipeline" and node.children:
            # This branch is a nested pipeline, render it with dashed border
            return self._render_nested_pipeline(node, estimator_id_counter)

        html = ['<div class="sk-item">']

        # Branch label
        html.append('<div class="sk-label-container">')
        timing_badge = self._get_timing_badge(
            node.mean_duration, node.mean_duration_display
        )
        html.append('<div class="sk-label">')
        html.append(
            f'<span class="sk-label-text">{self._escape_html(node.name)}</span>'
        )
        html.append(timing_badge)
        html.append("</div>")
        html.append("</div>")

        # If this branch has nested steps (e.g., a Pipeline)
        if node.children:
            html.append('<div class="sk-serial">')
            for child in node.children:
                # Recursively handle nested structures properly
                if child.kind == "Pipeline" and child.children:
                    # Nested pipeline - render with dashed border
                    html.append(
                        self._render_nested_pipeline(child, estimator_id_counter)
                    )
                elif child.kind == "ColumnTransformer":
                    # Nested ColumnTransformer - render it
                    html.append(
                        self._render_column_transformer(child, estimator_id_counter)
                    )
                else:
                    # Regular estimator
                    html.append(self._render_estimator(child, estimator_id_counter))
            html.append("</div>")

        html.append("</div>")
        return "\n".join(html)

    def _render_nested_pipeline(
        self, node: ProfileNode, estimator_id_counter: dict[str, int]
    ) -> str:
        """Render a nested Pipeline."""
        html = ['<div class="sk-item sk-dashed-wrapped">']

        # Label
        html.append('<div class="sk-label-container">')
        timing_badge = self._get_timing_badge(
            node.mean_duration, node.mean_duration_display
        )
        html.append('<div class="sk-label">')
        html.append(
            f'<span class="sk-label-text">{self._escape_html(node.name)}</span>'
        )
        html.append(timing_badge)
        html.append("</div>")
        html.append("</div>")

        # Steps
        if node.children:
            html.append('<div class="sk-serial">')
            for child in node.children:
                html.append(self._render_node(child, estimator_id_counter))
            html.append("</div>")

        html.append("</div>")
        return "\n".join(html)

    def _render_estimator(
        self, node: ProfileNode, estimator_id_counter: dict[str, int]
    ) -> str:
        """Render a leaf estimator/transformer."""
        estimator_id_counter["count"] += 1

        html = ['<div class="sk-item">']
        html.append('<div class="sk-estimator">')

        timing_badge = self._get_timing_badge(
            node.mean_duration, node.mean_duration_display
        )
        html.append('<div class="sk-estimator-name">')
        html.append(f"<span>{self._escape_html(node.name)}</span>")
        html.append(timing_badge)
        html.append("</div>")

        html.append("</div>")
        html.append("</div>")
        return "\n".join(html)

    def _get_timing_badge(self, duration: float, display: str) -> str:
        """Get timing badge HTML with appropriate color."""
        if duration < 0.001:  # < 1ms
            css_class = "timing-fast"
        elif duration < 0.01:  # < 10ms
            css_class = "timing-normal"
        elif duration < 0.1:  # < 100ms
            css_class = "timing-slow"
        else:
            css_class = "timing-very-slow"

        return f'<span class="stripje-timing-badge {css_class}">⏱ {display}</span>'

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


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


# ==========================================
# Helper: Column data extraction
# ==========================================


class ColumnDataHandler:
    """Handles data preparation and column extraction for ColumnTransformer."""

    @staticmethod
    def prepare(data: Any) -> Any:
        """Convert Series to DataFrame if needed."""
        is_series = hasattr(data, "index") and not hasattr(data, "iloc")
        if is_series:
            import pandas as pd

            return pd.DataFrame([data])
        return data

    @staticmethod
    def extract_columns(data: Any, columns: Any) -> Any:
        """Extract specific columns from data."""
        # Handle pandas Series (single row from DataFrame)
        if hasattr(data, "index") and not hasattr(data, "iloc"):
            return data[columns]

        # Handle pandas DataFrame
        if hasattr(data, "iloc"):
            return data[columns]

        # Handle numpy array - columns should be indices
        if isinstance(data, np.ndarray):
            return data[columns] if data.ndim == 1 else data[:, columns]

        return data

    @classmethod
    def extract_and_prepare_single_row(cls, data: Any, columns: Any) -> Any:
        """Extract columns and convert to single row if needed."""
        col_data = cls.extract_columns(data, columns)
        if hasattr(col_data, "iloc") and len(col_data) == 1:
            return col_data.iloc[0]
        return col_data


def _coerce_single_sample(sample: Any) -> Any:
    """Return a single sample in a numpy-friendly format for compiled steps."""
    if isinstance(sample, np.ndarray):
        return sample
    if hasattr(sample, "to_numpy"):
        arr = sample.to_numpy()
        return arr
    if isinstance(sample, dict):
        return np.asarray(list(sample.values()))
    return np.asarray(sample)


def _determine_method(step: Any, is_last: bool, final_method: str) -> str:
    """Determine which method to call on a pipeline step."""
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


# ==========================================
# Component Profilers: Handle specific types
# ==========================================


class ComponentProfiler(ABC):
    """Base class for profiling specific component types."""

    @abstractmethod
    def profile(
        self,
        component: Any,
        name: str,
        data: Any,
        parent: ProfileNode,
        method: str,
        y: Any = None,
    ) -> Any:
        """Profile in batch mode."""
        pass

    @abstractmethod
    def profile_compiled(
        self, component: Any, name: str, data: Any, parent: ProfileNode
    ) -> Any:
        """Profile in compiled mode."""
        pass


class DefaultComponentProfiler(ComponentProfiler):
    """Profiles regular transformers/estimators."""

    def profile(
        self,
        component: Any,
        name: str,
        data: Any,
        parent: ProfileNode,
        method: str,
        y: Any = None,
    ) -> Any:
        node = parent.child(name, type(component).__name__, method)

        start = perf_counter_ns()
        args = (data,) if method != "fit" else (data, y)
        result = getattr(component, method)(*args)
        end = perf_counter_ns()

        node.add_event(start, end)
        return result

    def profile_compiled(
        self, component: Any, name: str, data: Any, parent: ProfileNode
    ) -> Any:
        node = parent.child(name, type(component).__name__, "call")

        start = perf_counter_ns()
        handler = get_handler(type(component))
        if handler is None:
            handler = create_fallback_handler
        compiled_fn = handler(component)
        result = compiled_fn(data)
        result = _coerce_single_sample(result)
        end = perf_counter_ns()

        node.add_event(start, end)
        return result


class PipelineComponentProfiler(ComponentProfiler):
    """Profiles nested Pipelines (recursive)."""

    def __init__(self, batch_strategy: Any = None, compiled_strategy: Any = None):
        self.batch_strategy = batch_strategy
        self.compiled_strategy = compiled_strategy

    def profile(
        self,
        component: Pipeline,
        name: str,
        data: Any,
        parent: ProfileNode,
        method: str,
        _y: Any = None,
    ) -> Any:
        node = parent.child(name, type(component).__name__, method)
        start = perf_counter_ns()
        result = self.batch_strategy._profile_steps(
            component.steps, data, node, method, _y
        )
        end = perf_counter_ns()
        node.add_event(start, end)
        return result

    def profile_compiled(
        self, component: Pipeline, name: str, data: Any, parent: ProfileNode
    ) -> Any:
        node = parent.child(name, type(component).__name__, "call")
        start = perf_counter_ns()
        current = data
        for sub_name, sub_step in component.steps:
            current = self.compiled_strategy._profile_step(
                sub_step, sub_name, current, node
            )
        end = perf_counter_ns()
        node.add_event(start, end)
        return current


class BatchColumnTransformerProfiler(ComponentProfiler):
    """Handles ColumnTransformer in batch mode with MethodTimer."""

    def __init__(self, batch_strategy: Any = None):
        self.batch_strategy = batch_strategy

    def profile(
        self,
        transformer: ColumnTransformer,
        name: str,
        data: Any,
        parent: ProfileNode,
        method: str,
        _y: Any = None,
    ) -> Any:
        node = parent.child(name, type(transformer).__name__, method)

        if method not in {"transform", "fit_transform"}:
            start = perf_counter_ns()
            result = getattr(transformer, method)(data)
            end = perf_counter_ns()
            node.add_event(start, end)
            return result

        start = perf_counter_ns()
        with ExitStack() as stack:
            for trans_name, trans, columns in transformer.transformers_:
                self._instrument_sub_transformer(
                    trans_name, trans, columns, node, method, stack
                )

            with parallel_backend("threading"):
                result = getattr(transformer, method)(data)
        end = perf_counter_ns()
        node.add_event(start, end)

        # Add noop events for transformers that weren't actually called
        for child in node.children:
            if child.parent is node and not child.events:
                noop = perf_counter_ns()
                child.add_event(noop, noop)

        return result

    def _instrument_sub_transformer(
        self,
        name: str,
        trans: Any,
        columns: Any,
        parent: ProfileNode,
        method: str,
        stack: ExitStack,
    ) -> None:
        """Setup MethodTimer for sub-transformers."""
        metadata = {"columns": columns}

        if trans == "drop":
            child = parent.child(
                name=name, kind="drop", method="skip", metadata=metadata
            )
            if not child.events:
                noop = perf_counter_ns()
                child.add_event(noop, noop)
            return

        if trans == "passthrough":
            child = parent.child(
                name=name, kind="passthrough", method="identity", metadata=metadata
            )
            if not child.events:
                noop = perf_counter_ns()
                child.add_event(noop, noop)
            return

        child_method = method if hasattr(trans, method) else "transform"
        child = parent.child(
            name=name, kind=type(trans).__name__, method=child_method, metadata=metadata
        )

        if hasattr(trans, "steps"):
            stack.enter_context(MethodTimer(trans, child_method, child))
            self._instrument_pipeline_steps(trans, child, child_method, stack)
        else:
            stack.enter_context(MethodTimer(trans, child_method, child))

    def _instrument_pipeline_steps(
        self,
        pipeline: Any,
        parent_node: ProfileNode,
        final_method: str,
        stack: ExitStack,
    ) -> None:
        """Recursively instrument nested pipeline steps."""
        if not hasattr(pipeline, "steps"):
            return

        steps_list = list(pipeline.steps)
        for idx, (name, step) in enumerate(steps_list):
            method = _determine_method(step, idx == len(steps_list) - 1, final_method)
            metadata = {"method": method}
            step_node = parent_node.child(
                name=name, kind=type(step).__name__, method=method, metadata=metadata
            )

            if isinstance(step, Pipeline):
                stack.enter_context(MethodTimer(step, method, step_node))
                self._instrument_pipeline_steps(step, step_node, method, stack)
            elif isinstance(step, ColumnTransformer):
                stack.enter_context(MethodTimer(step, method, step_node))
            else:
                stack.enter_context(MethodTimer(step, method, step_node))

    def profile_compiled(
        self, transformer: ColumnTransformer, name: str, data: Any, parent: ProfileNode
    ) -> Any:
        # Not used in batch strategy
        raise NotImplementedError(
            "Use CompiledColumnTransformerProfiler for compiled mode"
        )


class CompiledColumnTransformerProfiler(ComponentProfiler):
    """Handles ColumnTransformer in compiled mode - extracts columns, profiles each."""

    def __init__(self, compiled_strategy: Any = None):
        self.compiled_strategy = compiled_strategy
        self.data_handler = ColumnDataHandler()

    def profile(
        self,
        transformer: ColumnTransformer,
        name: str,
        data: Any,
        parent: ProfileNode,
        method: str,
        y: Any = None,
    ) -> Any:
        # Not used in compiled strategy
        raise NotImplementedError("Use BatchColumnTransformerProfiler for batch mode")

    def profile_compiled(
        self, transformer: ColumnTransformer, name: str, data: Any, parent: ProfileNode
    ) -> Any:
        node = parent.child(name, type(transformer).__name__, "call")

        start = perf_counter_ns()
        data_df = self.data_handler.prepare(data)
        results = []

        for trans_name, trans, columns in transformer.transformers_:
            result = self._profile_sub_transformer(
                trans_name, trans, columns, data_df, node
            )
            if result is not None:
                results.append(result)

        combined = self._combine_results(results, data)
        end = perf_counter_ns()
        node.add_event(start, end)

        return combined

    def _profile_sub_transformer(
        self, name: str, trans: Any, columns: Any, data_df: Any, parent: ProfileNode
    ) -> Any | None:
        """Profile individual transformer within ColumnTransformer."""
        if trans == "drop":
            metadata = {"columns": columns}
            child = parent.child(
                name=name, kind="drop", method="skip", metadata=metadata
            )
            noop = perf_counter_ns()
            child.add_event(noop, noop)
            return None

        if trans == "passthrough":
            return self._profile_passthrough(name, columns, data_df, parent)

        if isinstance(trans, Pipeline):
            return self._profile_pipeline(name, trans, columns, data_df, parent)

        return self._profile_regular(name, trans, columns, data_df, parent)

    def _profile_passthrough(
        self, name: str, columns: Any, data_df: Any, parent: ProfileNode
    ) -> Any:
        """Profile a 'passthrough' transformer."""
        metadata = {"columns": columns}
        child = parent.child(
            name=name, kind="passthrough", method="identity", metadata=metadata
        )
        start = perf_counter_ns()
        col_data = self.data_handler.extract_and_prepare_single_row(data_df, columns)
        result = _coerce_single_sample(col_data)
        end = perf_counter_ns()
        child.add_event(start, end)
        return result

    def _profile_pipeline(
        self,
        name: str,
        trans: Pipeline,
        columns: Any,
        data_df: Any,
        parent: ProfileNode,
    ) -> Any:
        """Profile a nested Pipeline transformer."""
        metadata = {"columns": columns}
        child = parent.child(
            name=name, kind=type(trans).__name__, method="call", metadata=metadata
        )
        start = perf_counter_ns()

        col_data = self.data_handler.extract_and_prepare_single_row(data_df, columns)
        current = col_data
        for sub_name, sub_step in trans.steps:
            current = self.compiled_strategy._profile_step(
                sub_step, sub_name, current, child
            )

        result = _coerce_single_sample(current)
        end = perf_counter_ns()
        child.add_event(start, end)
        return result

    def _profile_regular(
        self, name: str, trans: Any, columns: Any, data_df: Any, parent: ProfileNode
    ) -> Any:
        """Profile a regular (non-Pipeline) transformer."""
        metadata = {"columns": columns}
        child = parent.child(
            name=name, kind=type(trans).__name__, method="call", metadata=metadata
        )
        start = perf_counter_ns()

        col_data = self.data_handler.extract_and_prepare_single_row(data_df, columns)
        handler = get_handler(type(trans))
        if handler is None:
            handler = create_fallback_handler
        compiled_fn = handler(trans)
        result = compiled_fn(_coerce_single_sample(col_data))
        result = _coerce_single_sample(result)

        end = perf_counter_ns()
        child.add_event(start, end)
        return result

    @staticmethod
    def _combine_results(results: list[Any], data: Any) -> Any:
        """Combine results from multiple transformers into a single array."""
        if results:
            combined = np.concatenate([r.ravel() if r.ndim > 1 else r for r in results])
            return combined
        return data


# ==========================================
# Strategy Pattern: Separate profiling modes
# ==========================================


class ProfilingStrategy(ABC):
    """Abstract base for different profiling strategies."""

    @abstractmethod
    def profile(
        self, pipeline: Pipeline, data: Any, parent: ProfileNode, y: Any = None
    ) -> Any:
        """Profile the pipeline and return output."""
        pass


class BatchProfilingStrategy(ProfilingStrategy):
    """Profiles sklearn pipelines with batch data (transform/fit/predict)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.component_profilers: dict[type, ComponentProfiler] = {}
        self._setup_profilers()

    def _setup_profilers(self) -> None:
        """Initialize component profilers with references to this strategy."""
        pipeline_profiler = PipelineComponentProfiler(batch_strategy=self)
        self.component_profilers = {
            Pipeline: pipeline_profiler,
            ColumnTransformer: BatchColumnTransformerProfiler(batch_strategy=self),
        }

    def profile(
        self, pipeline: Pipeline, data: Any, parent: ProfileNode, y: Any = None
    ) -> Any:
        return self._profile_steps(pipeline.steps, data, parent, self.mode, y)

    def _profile_steps(
        self,
        steps: Iterable[tuple[str, Any]],
        data: Any,
        parent: ProfileNode,
        final_method: str,
        y: Any,
    ) -> Any:
        """Profile a sequence of pipeline steps."""
        current = data
        steps_list = list(steps)

        for idx, (name, step) in enumerate(steps_list):
            is_last = idx == len(steps_list) - 1
            method = _determine_method(step, is_last, final_method)

            profiler = self._get_profiler_for(step)
            current = profiler.profile(step, name, current, parent, method, y)

        return current

    def _get_profiler_for(self, step: Any) -> ComponentProfiler:
        """Return appropriate profiler for component type."""
        for step_type, profiler in self.component_profilers.items():
            if isinstance(step, step_type):
                return profiler
        return DefaultComponentProfiler()


class CompiledProfilingStrategy(ProfilingStrategy):
    """Profiles compiled single-row pipelines."""

    def __init__(self) -> None:
        self.component_profilers: dict[type, ComponentProfiler] = {}
        self._setup_profilers()

    def _setup_profilers(self) -> None:
        """Initialize component profilers with references to this strategy."""
        pipeline_profiler = PipelineComponentProfiler(compiled_strategy=self)
        self.component_profilers = {
            Pipeline: pipeline_profiler,
            ColumnTransformer: CompiledColumnTransformerProfiler(
                compiled_strategy=self
            ),
        }

    def profile(
        self, pipeline: Pipeline, data: Any, parent: ProfileNode, _y: Any = None
    ) -> Any:
        current = data
        for name, step in pipeline.steps:
            current = self._profile_step(step, name, current, parent)
        return current

    def _profile_step(
        self, step: Any, name: str, data: Any, parent: ProfileNode
    ) -> Any:
        """Profile a single step in compiled mode."""
        profiler = self._get_profiler_for(step)
        return profiler.profile_compiled(step, name, data, parent)

    def _get_profiler_for(self, step: Any) -> ComponentProfiler:
        """Return appropriate profiler for component type."""
        for step_type, profiler in self.component_profilers.items():
            if isinstance(step, step_type):
                return profiler
        return DefaultComponentProfiler()


# ==========================================
# Main Profiler: Orchestrates everything
# ==========================================


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
        strategy = BatchProfilingStrategy(mode=self.mode)

        # Warmup
        for _ in range(self.warmup):
            strategy.profile(
                self.pipeline, X, ProfileNode("_warmup", "Pipeline", self.mode), y
            )

        # Profile
        root = ProfileNode(
            name="pipeline", kind=type(self.pipeline).__name__, method=self.mode
        )
        output: Any = None
        for _ in range(self.repetitions):
            start = perf_counter_ns()
            output = strategy.profile(self.pipeline, X, root, y)
            end = perf_counter_ns()
            root.add_event(start, end)

        return ProfileReport(root=root, output=output)

    def run_compiled(self, sample: Any) -> ProfileReport:
        """Profile the compiled single-row pipeline produced by Stripje."""
        strategy = CompiledProfilingStrategy()

        root = ProfileNode(name="compiled_pipeline", kind="callable", method="call")
        start = perf_counter_ns()
        output = strategy.profile(self.pipeline, sample, root)
        end = perf_counter_ns()
        root.add_event(start, end)

        return ProfileReport(root=root, output=output)
