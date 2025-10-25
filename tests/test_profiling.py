import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import stripje.profiling
from stripje.profiling import PipelineProfiler, ProfileNode, ProfileReport


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


def test_profile_report_to_dict():
    root = ProfileNode("root", "Pipeline", "predict")
    root.child("step1", "Transformer", "transform")
    report = ProfileReport(root)
    report_dict = report.to_dict()
    assert report_dict["name"] == "root"
    assert len(report_dict["children"]) == 1
    assert report_dict["children"][0]["name"] == "step1"


def test_profile_report_repr_html():
    X = np.ones((4, 3))
    pipeline = Pipeline(
        [
            ("sleep_a", SleepTransformer(0.01)),
            ("sleep_b", SleepTransformer(0.02)),
        ]
    )
    pipeline.fit(X)

    profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
    report = profiler.run(X)
    html = report._repr_html_()

    assert "stripje-profile-container" in html
    assert "sleep_a" in html
    assert "sleep_b" in html
    assert "⏱" in html


def test_profiler_with_warmup_and_repetitions():
    X = np.ones((4, 3))
    pipeline = Pipeline([("sleep", SleepTransformer(0.01))])
    pipeline.fit(X)

    profiler = PipelineProfiler(pipeline, mode="transform", repetitions=3, warmup=2)
    report = profiler.run(X)

    assert report.root.call_count == 3
    assert report.root.children[0].call_count == 3


def test_compiled_profiler_with_warmup_and_repetitions():
    X = np.ones((4, 3))
    pipeline = Pipeline([("sleep", SleepTransformer(0.01))])
    pipeline.fit(X)

    # This is a bit of a trick to test the warmup and repetitions
    # on the compiled path, as it doesn't directly support it.
    # We can check the number of events on the nodes.
    _profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1, warmup=0)

    # manually call it multiple times
    root = ProfileNode(name="compiled_pipeline", kind="callable", method="call")
    strategy = stripje.profiling.CompiledProfilingStrategy()
    for _ in range(2):  # warmup
        strategy.profile(pipeline, X[0], ProfileNode("_warmup", "Pipeline", "call"))

    for _ in range(3):  # repetitions
        strategy.profile(pipeline, X[0], root)

    assert root.children[0].call_count == 3
