import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import stripje.profiling
from stripje.profiling import CallEvent, PipelineProfiler, ProfileNode, ProfileReport


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


# ============================================================================
# Extended Coverage Tests
# ============================================================================


class QuickTransformer(TransformerMixin, BaseEstimator):
    """Simple transformer for testing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        time.sleep(0.001)
        return np.asarray(X)


class QuickEstimator(BaseEstimator):
    """Simple estimator for testing."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        time.sleep(0.001)
        return np.ones(len(X))


class TestCallEvent:
    """Tests for CallEvent class."""

    def test_call_event_duration(self):
        """Test CallEvent duration calculation."""
        start = 1_000_000_000  # 1 second in nanoseconds
        end = 2_500_000_000  # 2.5 seconds
        event = CallEvent(start_ns=start, end_ns=end)

        assert event.duration_seconds == pytest.approx(1.5, abs=0.001)


class TestProfileNode:
    """Tests for ProfileNode class."""

    def test_profile_node_child_creation(self):
        """Test creating child nodes."""
        root = ProfileNode("root", "Pipeline", "transform")
        child1 = root.child("step1", "Transformer", "transform")
        child2 = root.child("step2", "Estimator", "predict")

        assert len(root.children) == 2
        assert child1.name == "step1"
        assert child2.name == "step2"
        assert child1.parent is root
        assert child2.parent is root

    def test_profile_node_child_reuse(self):
        """Test that getting same child returns existing node."""
        root = ProfileNode("root", "Pipeline", "transform")
        child1 = root.child("step1", "Transformer", "transform", {"a": 1})
        child2 = root.child("step1", "Transformer", "transform", {"b": 2})

        assert child1 is child2
        assert len(root.children) == 1
        assert child1.metadata == {"a": 1, "b": 2}

    def test_profile_node_add_event(self):
        """Test adding timing events to a node."""
        node = ProfileNode("test", "Transformer", "transform")
        node.add_event(0, 1_000_000_000)  # 1 second
        node.add_event(0, 2_000_000_000)  # 2 seconds

        assert node.call_count == 2
        assert node.last_duration == pytest.approx(2.0, abs=0.001)
        assert node.mean_duration == pytest.approx(1.5, abs=0.001)

    def test_profile_node_empty_events(self):
        """Test node with no events."""
        node = ProfileNode("test", "Transformer", "transform")

        assert node.call_count == 0
        assert node.last_duration == 0.0
        assert node.mean_duration == 0.0
        assert "0.000" in node.last_duration_display
        assert "0.000" in node.mean_duration_display


class TestProfileReport:
    """Tests for ProfileReport class."""

    def test_profile_report_to_dict_extended(self):
        """Test converting profile report to dictionary with metadata."""
        root = ProfileNode("pipeline", "Pipeline", "predict")
        root.add_event(0, 1_000_000_000)

        step1 = root.child("step1", "Transformer", "transform", {"feature": "value"})
        step1.add_event(0, 500_000_000)

        step2 = root.child("step2", "Estimator", "predict")
        step2.add_event(0, 500_000_000)

        report = ProfileReport(root, output=[1, 2, 3])
        result = report.to_dict()

        assert result["name"] == "pipeline"
        assert result["kind"] == "Pipeline"
        assert result["method"] == "predict"
        assert result["call_count"] == 1
        assert len(result["children"]) == 2
        assert result["children"][0]["name"] == "step1"
        assert result["children"][0]["metadata"] == {"feature": "value"}

    def test_profile_report_with_output(self):
        """Test profile report storing output."""
        root = ProfileNode("test", "Pipeline", "predict")
        output = np.array([1, 2, 3])
        report = ProfileReport(root, output=output)

        assert report.output is not None
        np.testing.assert_array_equal(report.output, output)


class TestPipelineProfilerModes:
    """Test PipelineProfiler with different modes."""

    def test_profiler_predict_mode(self):
        """Test profiler in predict mode."""
        X = np.ones((5, 3))
        y = np.array([0, 1, 0, 1, 0])

        pipeline = Pipeline(
            [
                ("transformer", QuickTransformer()),
                ("estimator", QuickEstimator()),
            ]
        )
        pipeline.fit(X, y)

        profiler = PipelineProfiler(pipeline, mode="predict", repetitions=1)
        report = profiler.run(X)

        assert report.root.name == "pipeline"
        assert report.root.method == "predict"
        assert len(report.root.children) == 2
        assert report.output is not None

    def test_profiler_transform_mode(self):
        """Test profiler in transform mode."""
        X = np.ones((5, 3))

        pipeline = Pipeline(
            [
                ("transformer1", QuickTransformer()),
                ("transformer2", QuickTransformer()),
            ]
        )
        pipeline.fit(X)

        profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
        report = profiler.run(X)

        assert report.root.method == "transform"
        assert len(report.root.children) == 2

    def test_profiler_with_column_transformer(self):
        """Test profiler with ColumnTransformer."""
        X = np.ones((5, 4))
        y = np.array([0, 1, 0, 1, 0])

        ct = ColumnTransformer(
            [
                ("scale1", StandardScaler(), [0, 1]),
                ("scale2", StandardScaler(), [2, 3]),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", ct),
                ("classifier", LogisticRegression()),
            ]
        )
        pipeline.fit(X, y)

        profiler = PipelineProfiler(pipeline, mode="predict", repetitions=1)
        report = profiler.run(X)

        # Check that ColumnTransformer is profiled
        assert report.root.name == "pipeline"
        assert any(child.kind == "ColumnTransformer" for child in report.root.children)

    def test_profiler_with_nested_pipeline(self):
        """Test profiler with nested Pipeline."""
        X = np.ones((5, 3))

        inner_pipeline = Pipeline(
            [
                ("inner_trans1", QuickTransformer()),
                ("inner_trans2", QuickTransformer()),
            ]
        )

        outer_pipeline = Pipeline(
            [
                ("nested", inner_pipeline),
                ("outer_trans", QuickTransformer()),
            ]
        )
        outer_pipeline.fit(X)

        profiler = PipelineProfiler(outer_pipeline, mode="transform", repetitions=1)
        report = profiler.run(X)

        # Check nested structure
        assert len(report.root.children) == 2
        nested_node = report.root.children[0]
        assert nested_node.name == "nested"
        assert len(nested_node.children) == 2

    def test_profiler_multiple_repetitions(self):
        """Test profiler with multiple repetitions."""
        X = np.ones((5, 3))

        pipeline = Pipeline([("transformer", QuickTransformer())])
        pipeline.fit(X)

        profiler = PipelineProfiler(pipeline, mode="transform", repetitions=5)
        report = profiler.run(X)

        assert report.root.call_count == 5
        assert report.root.children[0].call_count == 5
        assert len(report.root.events) == 5


class TestCompiledProfiling:
    """Test profiling of compiled pipelines."""

    def test_compiled_profiler_single_row(self):
        """Test compiled profiler with single row input."""
        X = np.ones((5, 3))

        pipeline = Pipeline(
            [
                ("trans1", QuickTransformer()),
                ("trans2", QuickTransformer()),
            ]
        )
        pipeline.fit(X)

        profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
        report = profiler.run_compiled(X[0])

        assert report.root.name == "compiled_pipeline"
        assert len(report.root.children) == 2

    def test_compiled_profiler_with_list_input(self):
        """Test compiled profiler with list input."""
        X = np.ones((5, 3))

        pipeline = Pipeline([("transformer", QuickTransformer())])
        pipeline.fit(X)

        profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
        list_input = [1.0, 2.0, 3.0]
        report = profiler.run_compiled(list_input)

        assert report.root.name == "compiled_pipeline"


class TestHTMLRendering:
    """Test HTML rendering features."""

    def test_html_with_column_transformer(self):
        """Test HTML rendering with ColumnTransformer."""
        X = np.ones((5, 4))
        y = np.array([0, 1, 0, 1, 0])

        ct = ColumnTransformer(
            [
                ("num1", StandardScaler(), [0, 1]),
                ("num2", StandardScaler(), [2, 3]),
            ]
        )

        pipeline = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
        pipeline.fit(X, y)

        profiler = PipelineProfiler(pipeline, mode="predict", repetitions=1)
        report = profiler.run(X)
        html = report._repr_html_()

        assert "preprocessor" in html
        assert "num1" in html
        assert "num2" in html
        assert "sk-parallel" in html  # ColumnTransformer uses parallel layout

    def test_html_with_nested_pipeline(self):
        """Test HTML rendering with nested Pipeline."""
        X = np.ones((5, 3))

        inner = Pipeline([("inner_trans", QuickTransformer())])
        outer = Pipeline([("nested", inner), ("outer_trans", QuickTransformer())])
        outer.fit(X)

        profiler = PipelineProfiler(outer, mode="transform", repetitions=1)
        report = profiler.run(X)
        html = report._repr_html_()

        assert "nested" in html
        assert "inner_trans" in html
        assert "outer_trans" in html

    def test_html_timing_displays(self):
        """Test that timing information is displayed in HTML."""
        X = np.ones((5, 3))

        pipeline = Pipeline([("transformer", QuickTransformer())])
        pipeline.fit(X)

        profiler = PipelineProfiler(pipeline, mode="transform", repetitions=1)
        report = profiler.run(X)
        html = report._repr_html_()

        # Should contain timing badge/display
        assert "⏱" in html or "ms" in html or "timing" in html.lower()
