"""
Tests for stripje.profiling module.

Tests cover:
- ProfileResult dataclass functionality
- PipelineProfiler class functionality
- CompiledPipelineProfiler class functionality
- CompiledProfileResult dataclass functionality
- profile_pipeline_compilation function
- Various profiling scenarios and edge cases
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import from the actual module to test against real implementation
from stripje.profiling import (
    CompiledPipelineProfiler,
    CompiledProfileResult,
    PipelineProfiler,
    ProfileResult,
    profile_pipeline_compilation,
)


class TestProfileResult:
    """Test ProfileResult dataclass functionality."""

    def test_profile_result_creation(self):
        """Test basic ProfileResult creation."""
        result = ProfileResult(
            step_name="scaler",
            operation="fit",
            mean_time=1.5,
            std_time=0.1,
            min_time=1.4,
            max_time=1.7,
            num_runs=10,
            step_type="StandardScaler",
        )

        assert result.step_name == "scaler"
        assert result.operation == "fit"
        assert result.mean_time == 1.5
        assert result.std_time == 0.1
        assert result.min_time == 1.4
        assert result.max_time == 1.7
        assert result.num_runs == 10
        assert result.step_type == "StandardScaler"
        assert result.children == []

    def test_profile_result_with_children(self):
        """Test ProfileResult with children."""
        child = ProfileResult(
            step_name="inner_scaler",
            operation="fit",
            mean_time=0.5,
            std_time=0.05,
            min_time=0.45,
            max_time=0.55,
            num_runs=10,
            step_type="StandardScaler",
        )

        parent = ProfileResult(
            step_name="pipeline",
            operation="fit",
            mean_time=1.5,
            std_time=0.1,
            min_time=1.4,
            max_time=1.7,
            num_runs=10,
            step_type="Pipeline",
            children=[child],
        )

        assert len(parent.children) == 1
        assert parent.children[0] == child

    def test_format_time_microseconds(self):
        """Test time formatting in microseconds."""
        result = ProfileResult(
            step_name="test",
            operation="test",
            mean_time=0.0000015,  # 1.5 microseconds
            std_time=0.0000001,
            min_time=0.0000014,
            max_time=0.0000017,
            num_runs=10,
            step_type="Test",
        )

        time_val, unit = result._format_time(result.mean_time)
        assert unit == "μs"
        assert abs(time_val - 1.5) < 0.01

    def test_format_time_milliseconds(self):
        """Test time formatting in milliseconds."""
        result = ProfileResult(
            step_name="test",
            operation="test",
            mean_time=0.0015,  # 1.5 milliseconds
            std_time=0.0001,
            min_time=0.0014,
            max_time=0.0017,
            num_runs=10,
            step_type="Test",
        )

        time_val, unit = result._format_time(result.mean_time)
        assert unit == "ms"
        assert abs(time_val - 1.5) < 0.01

    def test_format_time_seconds(self):
        """Test time formatting in seconds."""
        result = ProfileResult(
            step_name="test",
            operation="test",
            mean_time=1.5,  # 1.5 seconds
            std_time=0.1,
            min_time=1.4,
            max_time=1.7,
            num_runs=10,
            step_type="Test",
        )

        time_val, unit = result._format_time(result.mean_time)
        assert unit == "s"
        assert abs(time_val - 1.5) < 0.01

    def test_format_time_minutes(self):
        """Test time formatting in minutes."""
        result = ProfileResult(
            step_name="test",
            operation="test",
            mean_time=90.0,  # 90 seconds = 1.5 minutes
            std_time=5.0,
            min_time=85.0,
            max_time=95.0,
            num_runs=10,
            step_type="Test",
        )

        time_val, unit = result._format_time(result.mean_time)
        assert unit == "min"
        assert abs(time_val - 1.5) < 0.01

    def test_format_time_nan(self):
        """Test time formatting with NaN values."""
        result = ProfileResult(
            step_name="test",
            operation="test",
            mean_time=float("nan"),
            std_time=float("nan"),
            min_time=float("nan"),
            max_time=float("nan"),
            num_runs=0,
            step_type="Test",
        )

        time_val, unit = result._format_time(result.mean_time)
        assert np.isnan(time_val)
        assert unit == "s"

    def test_get_formatted_times(self):
        """Test getting all formatted times."""
        result = ProfileResult(
            step_name="test",
            operation="test",
            mean_time=0.0015,  # 1.5 ms
            std_time=0.0001,  # 0.1 ms
            min_time=0.0014,  # 1.4 ms
            max_time=0.0017,  # 1.7 ms
            num_runs=10,
            step_type="Test",
        )

        formatted = result.get_formatted_times()

        assert "1.500ms" in formatted["mean"]
        # std_time could be either microseconds or milliseconds depending on precision
        assert "ms" in formatted["std"] or "μs" in formatted["std"]
        assert "1.400ms" in formatted["min"]
        assert "1.700ms" in formatted["max"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        child = ProfileResult(
            step_name="child",
            operation="fit",
            mean_time=0.5,
            std_time=0.05,
            min_time=0.45,
            max_time=0.55,
            num_runs=10,
            step_type="ChildType",
        )

        result = ProfileResult(
            step_name="parent",
            operation="fit",
            mean_time=1.5,
            std_time=0.1,
            min_time=1.4,
            max_time=1.7,
            num_runs=10,
            step_type="ParentType",
            children=[child],
        )

        result_dict = result.to_dict()

        assert result_dict["step_name"] == "parent"
        assert result_dict["operation"] == "fit"
        assert result_dict["mean_time"] == 1.5
        assert result_dict["step_type"] == "ParentType"
        assert "mean_time_formatted" in result_dict
        assert len(result_dict["children"]) == 1
        assert result_dict["children"][0]["step_name"] == "child"


class TestPipelineProfiler:
    """Test PipelineProfiler class functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        return X, y

    @pytest.fixture
    def sample_data_pandas(self):
        """Create sample pandas DataFrame data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y, name="target")
        return X_df, y_series

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple pipeline for testing."""
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )

    @pytest.fixture
    def column_transformer_pipeline(self):
        """Create a pipeline with ColumnTransformer for testing."""
        numeric_features = [0, 1, 2, 3, 4]
        categorical_features = [5, 6, 7, 8, 9]

        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False),
                    categorical_features,
                ),
            ]
        )

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )

    @pytest.fixture
    def nested_pipeline(self):
        """Create a nested pipeline for testing."""
        inner_pipeline = Pipeline([("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            [
                ("numeric", inner_pipeline, list(range(5))),
                ("passthrough", "passthrough", list(range(5, 10))),
            ]
        )

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
            ]
        )

    def test_profiler_initialization_defaults(self):
        """Test PipelineProfiler with default parameters."""
        profiler = PipelineProfiler()

        assert profiler.warmup_runs == 3
        assert profiler.profile_runs == 10
        assert profiler.cache_invalidation is True
        assert profiler.verbose is False
        assert profiler.results == {}

    def test_profiler_initialization_custom(self):
        """Test PipelineProfiler with custom parameters."""
        profiler = PipelineProfiler(
            warmup_runs=2, profile_runs=5, cache_invalidation=False, verbose=True
        )

        assert profiler.warmup_runs == 2
        assert profiler.profile_runs == 5
        assert profiler.cache_invalidation is False
        assert profiler.verbose is True

    def test_invalidate_caches(self):
        """Test cache invalidation functionality."""
        profiler = PipelineProfiler(cache_invalidation=True)

        # This should not raise an error
        profiler._invalidate_caches()

        # Test with cache invalidation disabled
        profiler_no_cache = PipelineProfiler(cache_invalidation=False)
        profiler_no_cache._invalidate_caches()  # Should do nothing

    def test_get_step_type(self):
        """Test step type identification."""
        profiler = PipelineProfiler()

        # Test Pipeline
        pipeline = Pipeline([("scaler", StandardScaler())])
        assert profiler._get_step_type(pipeline) == "Pipeline"

        # Test ColumnTransformer
        ct = ColumnTransformer([("num", StandardScaler(), [0])])
        assert profiler._get_step_type(ct) == "ColumnTransformer"

        # Test regular estimator
        scaler = StandardScaler()
        assert profiler._get_step_type(scaler) == "StandardScaler"

        # Test with object without __class__
        mock_obj = Mock()
        del mock_obj.__class__
        step_type = profiler._get_step_type(mock_obj)
        assert isinstance(step_type, str)

    def test_time_operation(self, simple_pipeline, sample_data):
        """Test the _time_operation method."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=3, verbose=False)

        def test_operation():
            return simple_pipeline.fit(X, y)

        times = profiler._time_operation(test_operation)

        assert len(times) <= profiler.profile_runs  # May be less due to filtering
        assert all(isinstance(t, float) for t in times)
        assert all(t >= 0 for t in times)

    def test_time_operation_with_exception(self):
        """Test _time_operation with operations that raise exceptions."""
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=3, verbose=False)

        def failing_operation():
            raise ValueError("Test error")

        times = profiler._time_operation(failing_operation)

        # Should return empty list when all operations fail
        assert times == []

    def test_profile_basic_fit(self, simple_pipeline, sample_data):
        """Test basic fit operation profiling."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(pipeline=simple_pipeline, X=X, y=y, operations="fit")

        assert "fit" in results
        result = results["fit"]
        assert result.operation == "fit"
        assert result.step_name == "root_pipeline"
        assert result.step_type == "Pipeline"
        assert result.mean_time > 0
        assert len(result.children) > 0  # Should have children for pipeline steps

    def test_profile_transform_operation(self, simple_pipeline, sample_data):
        """Test transform operation profiling."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(
            pipeline=simple_pipeline, X=X, y=y, operations="transform"
        )

        assert "transform" in results
        result = results["transform"]
        assert result.operation == "transform"
        # Note: transform operation might have 0 time if it's very fast or unsupported
        assert result.mean_time >= 0

    def test_profile_predict_operation(self, simple_pipeline, sample_data):
        """Test predict operation profiling."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(
            pipeline=simple_pipeline, X=X, y=y, operations="predict"
        )

        assert "predict" in results
        result = results["predict"]
        assert result.operation == "predict"
        assert result.mean_time > 0

    def test_profile_multiple_operations(self, simple_pipeline, sample_data):
        """Test profiling multiple operations."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            operations=["fit", "predict", "transform"],
        )

        assert len(results) == 3
        assert "fit" in results
        assert "predict" in results
        assert "transform" in results

        for op, result in results.items():
            assert result.operation == op
            assert result.mean_time >= 0  # Could be 0 for very fast operations

    def test_profile_column_transformer(self, column_transformer_pipeline, sample_data):
        """Test profiling of ColumnTransformer pipeline."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(
            pipeline=column_transformer_pipeline, X=X, y=y, operations="fit"
        )

        assert "fit" in results
        result = results["fit"]
        assert len(result.children) > 0  # Should have children for pipeline steps

        # Check that ColumnTransformer step has its own children
        preprocessor_child = None
        for child in result.children:
            if "preprocessor" in child.step_name:
                preprocessor_child = child
                break

        assert preprocessor_child is not None
        assert len(preprocessor_child.children) > 0  # Should have transformer children

    def test_profile_nested_pipeline(self, nested_pipeline, sample_data):
        """Test profiling of nested pipeline structures."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(pipeline=nested_pipeline, X=X, y=y, operations="fit")

        assert "fit" in results
        result = results["fit"]
        assert len(result.children) > 0

        # Should have deeply nested children
        def check_for_nested_children(result_node):
            if result_node.children:
                return True
            for child in result_node.children:
                if check_for_nested_children(child):
                    return True
            return False

        # At least one level of nesting should exist
        assert any(len(child.children) > 0 for child in result.children)

    def test_profile_single_row_numpy(self, simple_pipeline, sample_data):
        """Test single-row profiling with numpy arrays."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        # Use first 5 samples
        X_samples = X[:5]
        y_samples = y[:5] if y is not None else None

        results = profiler.profile_single_row(
            pipeline=simple_pipeline,
            X_samples=X_samples,
            y_samples=y_samples,
            operations="predict",
        )

        assert "predict" in results
        result = results["predict"]
        assert result.operation == "predict"
        assert result.step_name == "single_row_pipeline"
        assert result.num_runs > 0

    def test_profile_single_row_pandas(self, simple_pipeline, sample_data_pandas):
        """Test single-row profiling with pandas DataFrames."""
        X_df, y_series = sample_data_pandas
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        # Use first 5 samples
        X_samples = X_df.iloc[:5]
        y_samples = y_series.iloc[:5]

        results = profiler.profile_single_row(
            pipeline=simple_pipeline,
            X_samples=X_samples,
            y_samples=y_samples,
            operations="predict",
        )

        assert "predict" in results
        result = results["predict"]
        assert result.operation == "predict"
        assert result.num_runs > 0

    def test_profile_single_row_with_lists(self, simple_pipeline, sample_data):
        """Test single-row profiling with list inputs."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        # Convert to list of samples - properly reshape for 1D arrays
        X_samples = [X[i : i + 1] for i in range(3)]  # Keep 2D shape
        y_samples = [y[i : i + 1] for i in range(3)] if y is not None else None

        results = profiler.profile_single_row(
            pipeline=simple_pipeline,
            X_samples=X_samples,
            y_samples=y_samples,
            operations="predict",
        )

        assert "predict" in results
        result = results["predict"]
        assert result.num_runs > 0

    def test_profile_with_unsupported_operation(self, simple_pipeline, sample_data):
        """Test profiling with operations not supported by the estimator."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        # Create a mock step that doesn't have predict method
        mock_step = Mock()
        del mock_step.predict  # Remove predict method
        mock_step.fit = Mock()
        mock_step.transform = Mock(return_value=X)

        result = profiler._profile_step(
            step=mock_step,
            step_name="mock_step",
            operation="predict",
            X=X,
            y=y,
            is_fitted=True,
        )

        assert result.step_name == "mock_step"
        assert result.operation == "predict"
        assert result.num_runs == 0  # Should indicate unsupported

    def test_create_unsupported_result(self):
        """Test creation of unsupported operation results."""
        profiler = PipelineProfiler(verbose=False)

        result = profiler._create_unsupported_result(
            step_name="test_step",
            operation="test_op",
            step_type="TestType",
            reason="Test reason",
        )

        assert result.step_name == "test_step"
        assert result.operation == "test_op"
        assert result.step_type == "TestType"
        assert result.mean_time == 0.0
        assert result.num_runs == 0

    def test_create_failed_result(self):
        """Test creation of failed operation results."""
        profiler = PipelineProfiler(verbose=False)

        result = profiler._create_failed_result(
            step_name="test_step",
            operation="test_op",
            step_type="TestType",
            reason="Test failure",
        )

        assert result.step_name == "test_step"
        assert result.operation == "test_op"
        assert result.step_type == "TestType"
        assert np.isnan(result.mean_time)
        assert result.num_runs == 0

    def test_print_report(self, simple_pipeline, sample_data, capsys):
        """Test print report functionality."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(pipeline=simple_pipeline, X=X, y=y, operations="fit")

        profiler.print_report(results)

        captured = capsys.readouterr()
        assert "FIT OPERATION" in captured.out
        assert "root_pipeline" in captured.out

    def test_print_report_with_nan_values(self, capsys):
        """Test print report with NaN values."""
        profiler = PipelineProfiler()

        result = ProfileResult(
            step_name="failed_step",
            operation="fit",
            mean_time=float("nan"),
            std_time=float("nan"),
            min_time=float("nan"),
            max_time=float("nan"),
            num_runs=0,
            step_type="FailedType",
        )

        results = {"fit": result}
        profiler.print_report(results)

        captured = capsys.readouterr()
        assert "FAILED" in captured.out

    def test_to_dataframe(self, simple_pipeline, sample_data):
        """Test conversion of results to DataFrame."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(pipeline=simple_pipeline, X=X, y=y, operations="fit")

        df = profiler.to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        expected_columns = [
            "operation",
            "path",
            "step_name",
            "step_type",
            "mean_time",
            "std_time",
            "min_time",
            "max_time",
            "num_runs",
            "mean_time_formatted",
            "std_time_formatted",
            "min_time_formatted",
            "max_time_formatted",
        ]
        for col in expected_columns:
            assert col in df.columns

    def test_profile_with_verbose_output(self, simple_pipeline, sample_data, capsys):
        """Test profiling with verbose output."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=True)

        profiler.profile(pipeline=simple_pipeline, X=X, y=y, operations="fit")

        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should have verbose output

    def test_profile_fit_transform_operation(self, simple_pipeline, sample_data):
        """Test fit_transform operation profiling."""
        X, y = sample_data
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        results = profiler.profile(
            pipeline=simple_pipeline, X=X, y=y, operations="fit_transform"
        )

        assert "fit_transform" in results
        result = results["fit_transform"]
        assert result.operation == "fit_transform"
        assert result.mean_time >= 0


class TestCompiledProfileResult:
    """Test CompiledProfileResult dataclass functionality."""

    def test_compiled_profile_result_creation(self):
        """Test basic CompiledProfileResult creation."""
        result = CompiledProfileResult(
            step_name="compiled_pipeline",
            operation="predict",
            mean_time=0.8,
            std_time=0.05,
            min_time=0.75,
            max_time=0.85,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=1.5,
            compiled_time=0.8,
            speedup=1.875,
            original_std=0.1,
            compiled_std=0.05,
        )

        assert result.step_name == "compiled_pipeline"
        assert result.operation == "predict"
        assert result.original_time == 1.5
        assert result.compiled_time == 0.8
        assert result.speedup == 1.875
        assert result.original_std == 0.1
        assert result.compiled_std == 0.05

    def test_get_speedup_info_faster(self):
        """Test speedup info when compiled is faster."""
        result = CompiledProfileResult(
            step_name="test",
            operation="predict",
            mean_time=0.8,
            std_time=0.05,
            min_time=0.75,
            max_time=0.85,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=1.5,
            compiled_time=0.8,
            speedup=1.875,
            original_std=0.1,
            compiled_std=0.05,
        )

        speedup_info = result.get_speedup_info()

        assert speedup_info["speedup"] == "1.88x"
        assert "87.5%" in speedup_info["improvement"]
        assert (
            "improvement" not in speedup_info["improvement"]
            or "slower" not in speedup_info["improvement"]
        )

    def test_get_speedup_info_slower(self):
        """Test speedup info when compiled is slower."""
        result = CompiledProfileResult(
            step_name="test",
            operation="predict",
            mean_time=1.5,
            std_time=0.1,
            min_time=1.4,
            max_time=1.6,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=0.8,
            compiled_time=1.5,
            speedup=0.533,
            original_std=0.05,
            compiled_std=0.1,
        )

        speedup_info = result.get_speedup_info()

        assert speedup_info["speedup"] == "0.53x"
        assert "slower" in speedup_info["improvement"]

    def test_get_speedup_info_zero_compiled_time(self):
        """Test speedup info with zero compiled time."""
        result = CompiledProfileResult(
            step_name="test",
            operation="predict",
            mean_time=0.0,
            std_time=0.0,
            min_time=0.0,
            max_time=0.0,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=1.5,
            compiled_time=0.0,
            speedup=float("inf"),
            original_std=0.1,
            compiled_std=0.0,
        )

        speedup_info = result.get_speedup_info()

        assert (
            "inf" in speedup_info["speedup"].lower()
        )  # Should handle infinity (format may vary)


class TestCompiledPipelineProfiler:
    """Test CompiledPipelineProfiler class functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        return X, y

    @pytest.fixture
    def sample_data_pandas(self):
        """Create sample pandas DataFrame data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y, name="target")
        return X_df, y_series

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple fitted pipeline for testing."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )
        return pipeline

    def test_compiled_profiler_initialization_defaults(self):
        """Test CompiledPipelineProfiler initialization with defaults."""
        profiler = CompiledPipelineProfiler()

        assert profiler.warmup_runs == 5
        assert profiler.profile_runs == 50
        assert profiler.cache_invalidation is True
        assert profiler.verbose is False
        assert isinstance(profiler.base_profiler, PipelineProfiler)

    def test_compiled_profiler_initialization_custom(self):
        """Test CompiledPipelineProfiler initialization with custom parameters."""
        profiler = CompiledPipelineProfiler(
            warmup_runs=3, profile_runs=20, cache_invalidation=False, verbose=True
        )

        assert profiler.warmup_runs == 3
        assert profiler.profile_runs == 20
        assert profiler.cache_invalidation is False
        assert profiler.verbose is True

    def test_invalidate_caches(self):
        """Test cache invalidation for compiled profiler."""
        profiler = CompiledPipelineProfiler(cache_invalidation=True)

        # This should not raise an error
        profiler._invalidate_caches()

        # Test with cache invalidation disabled
        profiler_no_cache = CompiledPipelineProfiler(cache_invalidation=False)
        profiler_no_cache._invalidate_caches()  # Should do nothing

    def test_prepare_data_samples_pandas(self, sample_data_pandas):
        """Test preparing data samples from pandas DataFrame."""
        X_df, y_series = sample_data_pandas
        profiler = CompiledPipelineProfiler()

        samples = profiler._prepare_data_samples(X_df, num_samples=5)

        assert len(samples) == 5
        assert all(isinstance(sample, list) for sample in samples)
        assert len(samples[0]) == X_df.shape[1]

    def test_prepare_data_samples_numpy(self, sample_data):
        """Test preparing data samples from numpy array."""
        X, y = sample_data
        profiler = CompiledPipelineProfiler()

        samples = profiler._prepare_data_samples(X, num_samples=5)

        assert len(samples) == 5
        assert all(isinstance(sample, list) for sample in samples)
        assert len(samples[0]) == X.shape[1]

    def test_prepare_data_samples_1d_numpy(self):
        """Test preparing data samples from 1D numpy array."""
        X = np.array([1, 2, 3, 4, 5])
        profiler = CompiledPipelineProfiler()

        samples = profiler._prepare_data_samples(X, num_samples=3)

        assert len(samples) == 1  # Only one sample from 1D array
        assert samples[0] == X.tolist()

    def test_prepare_data_samples_list(self):
        """Test preparing data samples from list."""
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        profiler = CompiledPipelineProfiler()

        samples = profiler._prepare_data_samples(X, num_samples=2)

        assert len(samples) == 2
        assert samples == X[:2]

    def test_prepare_data_samples_unsupported_type(self):
        """Test preparing data samples with unsupported type."""
        profiler = CompiledPipelineProfiler()

        with pytest.raises(ValueError, match="Unsupported data type"):
            profiler._prepare_data_samples("unsupported_type")

    def test_time_compiled_operation(self):
        """Test timing of compiled function."""
        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=3, verbose=False
        )

        # Mock compiled function
        mock_compiled_fn = Mock(return_value=np.array([0, 1, 0]))
        data_samples = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        times = profiler._time_compiled_operation(mock_compiled_fn, data_samples)

        assert len(times) <= profiler.profile_runs
        assert all(isinstance(t, float) for t in times)
        assert all(t >= 0 for t in times)

        # Verify the mock was called
        assert mock_compiled_fn.call_count >= profiler.profile_runs

    def test_time_compiled_operation_with_exceptions(self):
        """Test timing compiled function that raises exceptions."""
        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=3, verbose=False
        )

        # Mock compiled function that always fails
        mock_compiled_fn = Mock(side_effect=ValueError("Test error"))
        data_samples = [[1, 2, 3]]

        times = profiler._time_compiled_operation(mock_compiled_fn, data_samples)

        assert times == []  # Should be empty due to failures

    @patch("stripje.fast_pipeline.compile_pipeline")
    def test_profile_compiled_vs_original(
        self, mock_compile_pipeline, simple_pipeline, sample_data
    ):
        """Test profiling compiled vs original pipeline."""
        X, y = sample_data

        # Fit the pipeline first
        simple_pipeline.fit(X, y)

        # Mock compiled function
        mock_compiled_fn = Mock(
            return_value=np.array([0] * 20)
        )  # Return appropriate size
        mock_compile_pipeline.return_value = mock_compiled_fn

        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=3, verbose=False
        )

        result = profiler.profile_compiled_vs_original(
            pipeline=simple_pipeline, X=X, y=y, operation="predict", num_samples=5
        )

        assert isinstance(result, CompiledProfileResult)
        assert result.operation == "predict"
        assert result.step_name == "compiled_pipeline"
        assert result.step_type == "CompiledPipeline"
        assert result.original_time > 0
        assert result.compiled_time >= 0
        assert result.num_runs > 0

    @patch("stripje.fast_pipeline.compile_pipeline")
    def test_profile_compiled_vs_original_pandas(
        self, mock_compile_pipeline, simple_pipeline, sample_data_pandas
    ):
        """Test profiling with pandas DataFrame input."""
        X_df, y_series = sample_data_pandas

        # Fit the pipeline first
        simple_pipeline.fit(X_df, y_series)

        # Mock compiled function
        mock_compiled_fn = Mock(return_value=np.array([0] * 20))
        mock_compile_pipeline.return_value = mock_compiled_fn

        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=3, verbose=False
        )

        result = profiler.profile_compiled_vs_original(
            pipeline=simple_pipeline,
            X=X_df,
            y=y_series,
            operation="predict",
            num_samples=5,
        )

        assert isinstance(result, CompiledProfileResult)
        assert result.operation == "predict"

    @patch("stripje.fast_pipeline.compile_pipeline")
    def test_profile_compiled_vs_original_transform(
        self, mock_compile_pipeline, simple_pipeline, sample_data
    ):
        """Test profiling transform operation."""
        X, y = sample_data

        # Fit the pipeline first
        simple_pipeline.fit(X, y)

        # Mock compiled function that returns transformed data
        mock_compiled_fn = Mock(
            return_value=X[:20]
        )  # Return appropriate transformed data
        mock_compile_pipeline.return_value = mock_compiled_fn

        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=3, verbose=False
        )

        result = profiler.profile_compiled_vs_original(
            pipeline=simple_pipeline, X=X, y=y, operation="transform", num_samples=5
        )

        assert isinstance(result, CompiledProfileResult)
        assert result.operation == "transform"

    def test_profile_compiled_only(self):
        """Test profiling only a compiled function."""
        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=3, verbose=False
        )

        # Mock compiled function
        mock_compiled_fn = Mock(return_value=np.array([0, 1, 0]))
        data_samples = [[1, 2, 3], [4, 5, 6]]

        result = profiler.profile_compiled_only(
            compiled_fn=mock_compiled_fn, data_samples=data_samples, operation="predict"
        )

        assert isinstance(result, ProfileResult)
        assert result.operation == "predict"
        assert result.step_name == "compiled_function"
        assert result.step_type == "CompiledFunction"
        assert result.mean_time >= 0

    def test_profile_compiled_only_failed(self):
        """Test profiling compiled function that fails."""
        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=3, verbose=False
        )

        # Mock compiled function that always fails
        mock_compiled_fn = Mock(side_effect=ValueError("Test error"))
        data_samples = [[1, 2, 3]]

        result = profiler.profile_compiled_only(
            compiled_fn=mock_compiled_fn, data_samples=data_samples, operation="predict"
        )

        assert isinstance(result, ProfileResult)
        assert np.isnan(result.mean_time)
        assert result.num_runs == 0

    def test_print_comparison_report(self, capsys):
        """Test printing comparison report."""
        profiler = CompiledPipelineProfiler()

        result = CompiledProfileResult(
            step_name="compiled_pipeline",
            operation="predict",
            mean_time=0.8,
            std_time=0.05,
            min_time=0.75,
            max_time=0.85,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=1.5,
            compiled_time=0.8,
            speedup=1.875,
            original_std=0.1,
            compiled_std=0.05,
        )

        profiler.print_comparison_report(result)

        captured = capsys.readouterr()
        assert "COMPILED PIPELINE COMPARISON" in captured.out
        assert "PREDICT" in captured.out
        assert "Original Pipeline:" in captured.out
        assert "Compiled Pipeline:" in captured.out
        assert "Speedup:" in captured.out

    def test_print_comparison_report_failed(self, capsys):
        """Test printing comparison report with failed results."""
        profiler = CompiledPipelineProfiler()

        result = CompiledProfileResult(
            step_name="compiled_pipeline",
            operation="predict",
            mean_time=float("nan"),
            std_time=float("nan"),
            min_time=float("nan"),
            max_time=float("nan"),
            num_runs=0,
            step_type="CompiledPipeline",
            original_time=float("nan"),
            compiled_time=float("nan"),
            speedup=float("nan"),
            original_std=float("nan"),
            compiled_std=float("nan"),
        )

        profiler.print_comparison_report(result)

        captured = capsys.readouterr()
        assert "Profiling failed" in captured.out

    def test_print_comparison_report_performance_categories(self, capsys):
        """Test different performance categories in comparison report."""
        profiler = CompiledPipelineProfiler()

        # Test excellent performance (>= 10x speedup)
        result_excellent = CompiledProfileResult(
            step_name="test",
            operation="predict",
            mean_time=0.1,
            std_time=0.01,
            min_time=0.09,
            max_time=0.11,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=1.0,
            compiled_time=0.1,
            speedup=10.0,
            original_std=0.1,
            compiled_std=0.01,
        )

        profiler.print_comparison_report(result_excellent)
        captured = capsys.readouterr()
        assert "EXCELLENT" in captured.out

        # Test very good performance (>= 5x speedup)
        result_very_good = CompiledProfileResult(
            step_name="test",
            operation="predict",
            mean_time=0.2,
            std_time=0.02,
            min_time=0.18,
            max_time=0.22,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=1.0,
            compiled_time=0.2,
            speedup=5.0,
            original_std=0.1,
            compiled_std=0.02,
        )

        profiler.print_comparison_report(result_very_good)
        captured = capsys.readouterr()
        assert "VERY GOOD" in captured.out

        # Test minimal performance (< 1.1x speedup)
        result_minimal = CompiledProfileResult(
            step_name="test",
            operation="predict",
            mean_time=0.95,
            std_time=0.05,
            min_time=0.9,
            max_time=1.0,
            num_runs=10,
            step_type="CompiledPipeline",
            original_time=1.0,
            compiled_time=0.95,
            speedup=1.05,
            original_std=0.1,
            compiled_std=0.05,
        )

        profiler.print_comparison_report(result_minimal)
        captured = capsys.readouterr()
        assert "MINIMAL" in captured.out


class TestProfilePipelineCompilation:
    """Test the profile_pipeline_compilation convenience function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        return X, y

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple fitted pipeline for testing."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )
        return pipeline

    @patch("stripje.fast_pipeline.compile_pipeline")
    def test_profile_pipeline_compilation_basic(
        self, mock_compile_pipeline, simple_pipeline, sample_data
    ):
        """Test basic usage of profile_pipeline_compilation function."""
        X, y = sample_data

        # Fit the pipeline first
        simple_pipeline.fit(X, y)

        # Mock compiled function
        mock_compiled_fn = Mock(return_value=np.array([0] * 20))
        mock_compile_pipeline.return_value = mock_compiled_fn

        result = profile_pipeline_compilation(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            operation="predict",
            num_samples=5,
            verbose=False,
        )

        assert isinstance(result, CompiledProfileResult)
        assert result.operation == "predict"
        assert result.step_name == "compiled_pipeline"
        assert result.step_type == "CompiledPipeline"

    @patch("stripje.fast_pipeline.compile_pipeline")
    def test_profile_pipeline_compilation_verbose(
        self, mock_compile_pipeline, simple_pipeline, sample_data, capsys
    ):
        """Test profile_pipeline_compilation with verbose output."""
        X, y = sample_data

        # Fit the pipeline first
        simple_pipeline.fit(X, y)

        # Mock compiled function
        mock_compiled_fn = Mock(return_value=np.array([0] * 20))
        mock_compile_pipeline.return_value = mock_compiled_fn

        result = profile_pipeline_compilation(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            operation="predict",
            num_samples=5,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should have verbose output
        assert isinstance(result, CompiledProfileResult)

    @patch("stripje.fast_pipeline.compile_pipeline")
    def test_profile_pipeline_compilation_transform(
        self, mock_compile_pipeline, simple_pipeline, sample_data
    ):
        """Test profile_pipeline_compilation with transform operation."""
        X, y = sample_data

        # Fit the pipeline first
        simple_pipeline.fit(X, y)

        # Mock compiled function for transform
        mock_compiled_fn = Mock(
            return_value=X[:20]
        )  # Return appropriate transformed data
        mock_compile_pipeline.return_value = mock_compiled_fn

        result = profile_pipeline_compilation(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            operation="transform",
            num_samples=5,
            verbose=False,
        )

        assert isinstance(result, CompiledProfileResult)
        assert result.operation == "transform"


# Integration tests
class TestProfilingIntegration:
    """Integration tests for profiling functionality."""

    def test_complete_profiling_workflow(self):
        """Test complete profiling workflow with real data."""
        # Create data and pipeline - ensure multiple classes
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )

        # Test basic profiling
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)
        results = profiler.profile(
            pipeline=pipeline, X=X, y=y, operations=["fit", "predict", "transform"]
        )

        assert len(results) == 3
        assert "fit" in results
        assert "predict" in results
        assert "transform" in results

        # Convert to DataFrame
        df = profiler.to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Test single-row profiling - ensure we have samples from both classes
        # Find indices of each class
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]

        # Take one sample from each class plus one more
        selected_indices = [class_0_indices[0], class_1_indices[0], class_0_indices[1]]
        X_samples = X[selected_indices]
        y_samples = y[selected_indices]

        single_row_results = profiler.profile_single_row(
            pipeline=pipeline,
            X_samples=X_samples,
            y_samples=y_samples,
            operations="predict",
        )

        assert "predict" in single_row_results

    def test_nested_pipeline_profiling_integration(self):
        """Test profiling of complex nested pipelines."""
        # Create data
        X, y = make_classification(n_samples=50, n_features=8, random_state=42)

        # Create nested pipeline
        numeric_preprocessor = Pipeline([("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_preprocessor, list(range(4))),
                ("passthrough", "passthrough", list(range(4, 8))),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=3, random_state=42)),
            ]
        )

        # Profile the nested pipeline
        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)
        results = profiler.profile(pipeline=pipeline, X=X, y=y, operations="fit")

        assert "fit" in results
        result = results["fit"]

        # Should have nested structure
        assert len(result.children) > 0

        # Check for preprocessor and classifier children
        child_names = [child.step_name for child in result.children]
        assert any("preprocessor" in name for name in child_names)
        assert any("classifier" in name for name in child_names)

    @patch("stripje.fast_pipeline.compile_pipeline")
    def test_compiled_profiling_integration(self, mock_compile_pipeline):
        """Test complete compiled profiling workflow."""
        # Create data and pipeline - ensure multiple classes
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )

        # Fit pipeline
        pipeline.fit(X, y)

        # Mock compiled function
        mock_compiled_fn = Mock(return_value=np.array([0] * 20))
        mock_compile_pipeline.return_value = mock_compiled_fn

        # Test compiled profiling
        profiler = CompiledPipelineProfiler(
            warmup_runs=1, profile_runs=2, verbose=False
        )
        result = profiler.profile_compiled_vs_original(
            pipeline=pipeline, X=X, y=y, operation="predict", num_samples=5
        )

        assert isinstance(result, CompiledProfileResult)
        assert result.operation == "predict"

        # Test the convenience function as well
        convenience_result = profile_pipeline_compilation(
            pipeline=pipeline,
            X=X,
            y=y,
            operation="predict",
            num_samples=5,
            verbose=False,
        )

        assert isinstance(convenience_result, CompiledProfileResult)

    def test_profiling_with_different_data_types(self):
        """Test profiling with different input data types."""
        # Create numpy data - ensure multiple classes
        X_numpy, y_numpy = make_classification(
            n_samples=30, n_features=4, n_classes=2, random_state=42
        )

        # Create pandas data
        X_pandas = pd.DataFrame(X_numpy, columns=[f"feature_{i}" for i in range(4)])
        y_pandas = pd.Series(y_numpy, name="target")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )

        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        # Test with numpy arrays
        numpy_results = profiler.profile(
            pipeline=pipeline, X=X_numpy, y=y_numpy, operations="fit"
        )

        assert "fit" in numpy_results

        # Test with pandas DataFrames
        pandas_results = profiler.profile(
            pipeline=pipeline, X=X_pandas, y=y_pandas, operations="fit"
        )

        assert "fit" in pandas_results

        # Both should produce valid results
        assert numpy_results["fit"].mean_time > 0
        assert pandas_results["fit"].mean_time > 0

    def test_error_handling_integration(self):
        """Test error handling in various scenarios."""
        X, y = make_classification(
            n_samples=30, n_features=4, n_classes=2, random_state=42
        )

        # Test with pipeline that has steps without required methods
        class FaultyEstimator:
            def fit(self, X, y):
                return self

            # Missing predict method

        faulty_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("faulty", FaultyEstimator())]
        )

        profiler = PipelineProfiler(warmup_runs=1, profile_runs=2, verbose=False)

        # This should handle the missing predict method gracefully
        results = profiler.profile(
            pipeline=faulty_pipeline, X=X, y=y, operations="predict"
        )

        assert "predict" in results
        # The overall pipeline should fail gracefully
        # but individual steps might have different behaviors


if __name__ == "__main__":
    pytest.main([__file__])
