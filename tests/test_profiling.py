import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stripje.profiling import PipelineProfiler


class SleepTransformer(TransformerMixin, BaseEstimator):
	"""Transformer that sleeps for a configurable duration before returning input."""

	def __init__(self, sleep_seconds: float) -> None:
		self.sleep_seconds = sleep_seconds

	def fit(self, X, y=None):  # noqa: D401 - standard sklearn fit signature
		return self

	def transform(self, X):
		time.sleep(self.sleep_seconds)
		if hasattr(X, "to_numpy"):
			return X.to_numpy()
		return np.asarray(X)


@dataclass
class SleepEstimator(BaseEstimator):
	"""Estimator that sleeps and echoes the mean of the inputs."""

	sleep_seconds: float

	def fit(self, X, y=None):  # noqa: D401 - standard sklearn fit signature
		return self

	def predict(self, X):
		time.sleep(self.sleep_seconds)
		X = np.asarray(X)
		return np.ones(X.shape[0])


def assert_duration(duration, expected, tolerance=0.015):
	assert duration == pytest.approx(expected, rel=0.2, abs=tolerance)


def test_pipeline_profiler_records_step_durations():
	X = np.ones((4, 3))
	pipeline = Pipeline(
		[
			("sleep_a", SleepTransformer(0.05)),
			("sleep_b", SleepTransformer(0.02)),
		]
	)
	pipeline.fit(X)

	profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
	report = profiler.run(X)

	root = report.root
	assert root.name == "pipeline"
	assert len(root.children) == 2

	first_step, second_step = root.children
	assert first_step.name == "sleep_a"
	assert second_step.name == "sleep_b"

	assert_duration(first_step.last_duration, 0.05)
	assert_duration(second_step.last_duration, 0.02)

	sequential_expected = first_step.last_duration + second_step.last_duration
	assert root.last_duration >= sequential_expected


def test_column_transformer_parallel_branch_timings():
	df = pd.DataFrame({"num1": [1, 2, 3], "num2": [4, 5, 6], "cat": ["a", "b", "c"]})

	slow_numeric = Pipeline(
		[
			("sleep", SleepTransformer(0.05)),
			("impute", SimpleImputer(strategy="median")),
			("scale", StandardScaler()),
		]
	)

	slow_categorical = Pipeline(
		[
			("sleep", SleepTransformer(0.03)),
			("impute", SimpleImputer(strategy="most_frequent")),
			("encode", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	column_transformer = ColumnTransformer(
		transformers=[
			("numeric", slow_numeric, ["num1", "num2"]),
			("categorical", slow_categorical, ["cat"]),
		],
		n_jobs=2,
		remainder="drop",
	)

	pipeline = Pipeline(
		[
			("columns", column_transformer),
			("final", SleepEstimator(0.01)),
		]
	)
	pipeline.fit(df, np.zeros(len(df)))

	profiler = PipelineProfiler(pipeline, mode="predict", repetitions=1)
	report = profiler.run(df)

	root = report.root
	column_node = next(child for child in root.children if child.name == "columns")
	final_node = next(child for child in root.children if child.name == "final")

	assert column_node.name == "columns"
	assert len(column_node.children) == 2
	numeric_node = next(child for child in column_node.children if child.name == "numeric")
	categorical_node = next(child for child in column_node.children if child.name == "categorical")

	assert_duration(numeric_node.last_duration, 0.05)
	assert_duration(categorical_node.last_duration, 0.03)

	numeric_child_names = [child.name for child in numeric_node.children]
	assert ["sleep", "impute", "scale"] == numeric_child_names
	numeric_sleep = next(child for child in numeric_node.children if child.name == "sleep")
	assert_duration(numeric_sleep.last_duration, 0.05)

	categorical_child_names = [child.name for child in categorical_node.children]
	assert ["sleep", "impute", "encode"] == categorical_child_names
	categorical_sleep = next(child for child in categorical_node.children if child.name == "sleep")
	assert_duration(categorical_sleep.last_duration, 0.03)

	assert column_node.last_duration >= max(
		numeric_node.last_duration, categorical_node.last_duration
	)
	assert column_node.last_duration <= (
		numeric_node.last_duration + categorical_node.last_duration + 0.02
	)

	assert_duration(final_node.last_duration, 0.01)


def test_compiled_pipeline_profiler_matches_step_structure():
	X = np.ones((2, 3))
	pipeline = Pipeline(
		[
			("sleep_a", SleepTransformer(0.04)),
			("sleep_b", SleepTransformer(0.02)),
		]
	)
	pipeline.fit(X)

	profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
	compiled_report = profiler.run_compiled(X[0])

	root = compiled_report.root
	assert root.name == "compiled_pipeline"
	assert [child.name for child in root.children] == ["sleep_a", "sleep_b"]

	first_step, second_step = root.children
	assert_duration(first_step.last_duration, 0.04)
	assert_duration(second_step.last_duration, 0.02)


def test_compiled_pipeline_profiler_handles_pandas_series():
	df = pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "c"])
	pipeline = Pipeline(
		[
			("sleep_a", SleepTransformer(0.03)),
			("sleep_b", SleepTransformer(0.01)),
		]
	)
	pipeline.fit(df.values)

	profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
	report = profiler.run_compiled(df.iloc[0])

	assert [child.name for child in report.root.children] == ["sleep_a", "sleep_b"]
	assert report.root.children[0].last_duration > 0

