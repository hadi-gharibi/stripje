"""Profiling utilities for scikit-learn pipelines and Stripje compiled pipelines."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter_ns
from types import MethodType
from typing import Any, Callable, Iterable, Optional

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
	parent: Optional["ProfileNode"] = field(default=None, repr=False)
	children: list["ProfileNode"] = field(default_factory=list)
	events: list[CallEvent] = field(default_factory=list)

	def child(
		self,
		name: str,
		kind: str,
		method: str,
		metadata: Optional[dict[str, Any]] = None,
	) -> "ProfileNode":
		"""Get or create a named child node."""
		for existing in self.children:
			if existing.name == name:
				if metadata:
					existing.metadata.update(metadata)
				existing.kind = kind
				existing.method = method
				return existing
		child = ProfileNode(name=name, kind=kind, method=method, metadata=metadata or {})
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
		self._bound_original: Optional[Callable[..., Any]] = None

	def __enter__(self) -> "MethodTimer":
		if not hasattr(self.obj, self.method_name):
			return self

		self._owned = self.method_name in vars(self.obj)
		if self._owned:
			self._original_attr = vars(self.obj)[self.method_name]
		self._bound_original = getattr(self.obj, self.method_name)

		@wraps(self._bound_original)
		def wrapper(_self, *args: Any, **kwargs: Any) -> Any:
			start = perf_counter_ns()
			try:
				return self._bound_original(*args, **kwargs)  # type: ignore[misc]
			finally:
				end = perf_counter_ns()
				self.node.add_event(start, end)

		bound_wrapper = MethodType(wrapper, self.obj)
		setattr(self.obj, self.method_name, bound_wrapper)
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		if not hasattr(self.obj, self.method_name):
			return
		if self._owned:
			setattr(self.obj, self.method_name, self._original_attr)
		else:
			try:
				delattr(self.obj, self.method_name)
			except AttributeError:
				pass


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

		root = ProfileNode(name="pipeline", kind=type(self.pipeline).__name__, method=self.mode)
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
		steps = self._compiled_steps()
		current = self._coerce_single_sample(sample)
		start_root = perf_counter_ns()

		for name, fn in steps:
			node = root.child(name=name, kind="compiled_step", method="call")
			start = perf_counter_ns()
			current = fn(current)
			current = self._coerce_single_sample(current)
			end = perf_counter_ns()
			node.add_event(start, end)

		end_root = perf_counter_ns()
		root.add_event(start_root, end_root)
		return ProfileReport(root=root, output=current)

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

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
		parent: Optional[ProfileNode],
		final_method: str,
		y: Any,
	) -> Any:
		current = data
		parent_node = parent or ProfileNode(name="pipeline", kind="Pipeline", method=final_method)

		steps_list = list(steps)
		for idx, (name, step) in enumerate(steps_list):
			is_last = idx == len(steps_list) - 1
			method = self._determine_method(step, is_last, final_method)
			metadata = {"method": method}
			node = parent_node.child(name=name, kind=type(step).__name__, method=method, metadata=metadata)

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
					child = node.child(name=name, kind="drop", method="skip", metadata=metadata)
					if not child.events:
						noop = perf_counter_ns()
						child.add_event(noop, noop)
					continue

				if trans == "passthrough":
					child = node.child(name=name, kind="passthrough", method="identity", metadata=metadata)
					if not child.events:
						noop = perf_counter_ns()
						child.add_event(noop, noop)
					continue

				child_method = method if hasattr(trans, method) else "transform"
				child = node.child(name=name, kind=type(trans).__name__, method=child_method, metadata=metadata)

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
			method = self._determine_method(step, idx == len(steps_list) - 1, final_method)
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
