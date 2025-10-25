"""
Comprehensive tests for the fast pipeline compiler.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from stripje.fast_pipeline import compile_pipeline, handle_column_transformer
from stripje.registry import get_supported_transformers


class TestPipelineCompiler:
    """Test suite for the pipeline compiler."""

    @pytest.fixture
    def mixed_data(self):
        """Generate mixed numeric and categorical data."""
        X, y = make_classification(
            n_samples=200, n_features=6, n_informative=4, n_redundant=1, random_state=42
        )

        # Convert to DataFrame and add categorical features
        df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
        df["cat_1"] = np.random.choice(["A", "B", "C"], size=len(df))
        df["cat_2"] = np.random.choice(["X", "Y"], size=len(df))

        return df, y

    @pytest.fixture
    def simple_numeric_data(self):
        """Generate simple numeric data."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return X, y

    def test_simple_pipeline_compilation(self, simple_numeric_data):
        """Test compilation of a simple pipeline."""
        X, y = simple_numeric_data

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        # Test multiple rows
        for i in range(10):
            test_row = X[i].tolist()
            original_pred = pipeline.predict([test_row])[0]
            fast_pred = fast_predict(test_row)

            assert original_pred == fast_pred, (
                f"Simple pipeline mismatch for row {i}: {original_pred} vs {fast_pred}"
            )

    def test_preprocessing_only_pipeline(self, simple_numeric_data):
        """Test pipeline with only preprocessing steps."""
        X, y = simple_numeric_data

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=3))]
        )

        pipeline.fit(X)
        fast_transform = compile_pipeline(pipeline)

        for i in range(10):
            test_row = X[i].tolist()
            original_result = pipeline.transform([test_row])[0]
            fast_result = fast_transform(test_row)

            assert np.allclose(original_result, fast_result, rtol=1e-10), (
                f"Preprocessing pipeline mismatch for row {i}"
            )

    def test_multi_step_preprocessing_pipeline(self, simple_numeric_data):
        """Test pipeline with multiple preprocessing steps."""
        X, y = simple_numeric_data

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selection", SelectKBest(score_func=f_classif, k=3)),
                ("pca", PCA(n_components=2)),
            ]
        )

        pipeline.fit(X, y)
        fast_transform = compile_pipeline(pipeline)

        for i in range(10):
            test_row = X[i].tolist()
            original_result = pipeline.transform([test_row])[0]
            fast_result = fast_transform(test_row)

            assert np.allclose(original_result, fast_result, rtol=1e-10), (
                f"Multi-step preprocessing pipeline mismatch for row {i}"
            )

    def test_column_transformer_numeric_only(self, simple_numeric_data):
        """Test ColumnTransformer with numeric features only."""
        X, y = simple_numeric_data

        # Create DataFrame for column names
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        preprocessor = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["feature_0", "feature_1", "feature_2"]),
                ("minmax", MinMaxScaler(), ["feature_3", "feature_4"]),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(df, y)
        fast_predict = compile_pipeline(pipeline)

        # Test with dictionary input (column names)
        for i in range(10):
            test_row_dict = df.iloc[i].to_dict()
            # Pass as DataFrame to maintain column names
            original_pred = pipeline.predict(df.iloc[[i]])[0]
            fast_pred = fast_predict(test_row_dict)

            assert original_pred == fast_pred, (
                f"ColumnTransformer (numeric) mismatch for row {i}"
            )

    def test_column_transformer_mixed_types(self, mixed_data):
        """Test ColumnTransformer with mixed data types."""
        df, y = mixed_data

        numeric_features = [col for col in df.columns if col.startswith("num_")]
        categorical_features = [col for col in df.columns if col.startswith("cat_")]

        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first"), categorical_features),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(df, y)
        fast_predict = compile_pipeline(pipeline)

        for i in range(10):
            test_row_dict = df.iloc[i].to_dict()
            # Pass as DataFrame to maintain column names
            original_pred = pipeline.predict(df.iloc[[i]])[0]
            fast_pred = fast_predict(test_row_dict)

            assert original_pred == fast_pred, (
                f"ColumnTransformer (mixed) mismatch for row {i}"
            )

    def test_column_transformer_passthrough(self, simple_numeric_data):
        """Test ColumnTransformer with passthrough."""
        X, y = simple_numeric_data
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        preprocessor = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["feature_0", "feature_1"]),
                ("passthrough", "passthrough", ["feature_2", "feature_3", "feature_4"]),
            ]
        )

        pipeline = Pipeline([("preprocessor", preprocessor)])

        pipeline.fit(df)
        fast_transform = compile_pipeline(pipeline)

        test_row_dict = df.iloc[0].to_dict()
        # Pass as DataFrame to maintain column names
        original_result = pipeline.transform(df.iloc[[0]])[0]
        fast_result = fast_transform(test_row_dict)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    def test_column_transformer_drop_columns(self, simple_numeric_data):
        """Test ColumnTransformer with dropped columns."""
        X, y = simple_numeric_data
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        preprocessor = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["feature_0", "feature_1", "feature_2"]),
                ("drop", "drop", ["feature_3", "feature_4"]),
            ]
        )

        pipeline = Pipeline([("preprocessor", preprocessor)])

        pipeline.fit(df)
        fast_transform = compile_pipeline(pipeline)

        test_row_dict = df.iloc[0].to_dict()
        # Pass as DataFrame to maintain column names
        original_result = pipeline.transform(df.iloc[[0]])[0]
        fast_result = fast_transform(test_row_dict)

        assert np.allclose(original_result, fast_result, rtol=1e-10)
        assert len(fast_result) == 3  # Only 3 features should remain

    def test_regression_pipeline(self):
        """Test compilation of regression pipeline."""
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = pipeline.predict([test_row])[0]
            fast_pred = fast_predict(test_row)

            assert np.isclose(original_pred, fast_pred, rtol=1e-10), (
                f"Regression pipeline mismatch for row {i}"
            )

    def test_unsupported_transformer_fallback(self):
        """Test that unsupported transformers use fallback mechanism."""

        X, y = make_classification(
            n_samples=50,
            n_features=3,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )

        # Create a pipeline with an unsupported transformer
        # (Note: PowerTransformer is actually supported, so we'll use a mock)
        class UnsupportedTransformer:
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("unsupported", UnsupportedTransformer())]
        )

        pipeline.fit(X, y)

        # This should now work with fallback mechanism instead of raising an error
        fast_pipeline = compile_pipeline(pipeline)

        # Test that it actually works
        test_row = X[0].tolist()
        orig_result = pipeline.transform([test_row])[0]
        fast_result = fast_pipeline(test_row)

        assert len(orig_result) == len(fast_result), (
            "Fallback mechanism should preserve output shape"
        )
        assert np.allclose(orig_result, fast_result), (
            "Fallback mechanism should produce same results"
        )

    def test_pipeline_with_different_input_formats(self, simple_numeric_data):
        """Test pipeline handles different input formats correctly."""
        X, y = simple_numeric_data

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        test_row = X[0]

        # Test with numpy array
        original_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row.tolist())
        assert original_pred == fast_pred

        # Test with list
        fast_pred = fast_predict(test_row.tolist())
        assert original_pred == fast_pred

        # Test with tuple
        fast_pred = fast_predict(tuple(test_row.tolist()))
        assert original_pred == fast_pred

    def test_pipeline_consistency_multiple_calls(self, simple_numeric_data):
        """Test that compiled pipeline is consistent across multiple calls."""
        X, y = simple_numeric_data

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        test_row = X[0].tolist()

        # Call multiple times
        predictions = [fast_predict(test_row) for _ in range(10)]

        # All predictions should be identical
        for pred in predictions[1:]:
            assert pred == predictions[0]

    def test_get_supported_transformers(self):
        """Test that get_supported_transformers returns expected types."""
        supported = get_supported_transformers()

        # Check that common transformers are supported
        transformer_names = [cls.__name__ for cls in supported]

        expected_transformers = [
            "StandardScaler",
            "MinMaxScaler",
            "LogisticRegression",
            "LinearRegression",
            "PCA",
            "SelectKBest",
        ]

        for transformer in expected_transformers:
            assert transformer in transformer_names, (
                f"{transformer} should be supported"
            )

    def test_column_transformer_handler_directly(self, mixed_data):
        """Test ColumnTransformer handler directly."""
        df, y = mixed_data

        numeric_features = [col for col in df.columns if col.startswith("num_")]
        categorical_features = [col for col in df.columns if col.startswith("cat_")]

        ct = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first"), categorical_features),
            ]
        )

        ct.fit(df, y)

        fast_ct = handle_column_transformer(ct)

        test_row_dict = df.iloc[0].to_dict()
        # Pass as DataFrame to maintain column names
        original_result = ct.transform(df.iloc[[0]])[0]
        fast_result = fast_ct(test_row_dict)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    def test_empty_pipeline_edge_case(self):
        """Test edge case with empty or minimal pipeline."""
        X, y = make_classification(
            n_samples=50,
            n_features=3,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )

        # Pipeline with just a classifier (no preprocessing)
        pipeline = Pipeline(
            [("classifier", LogisticRegression(random_state=42, max_iter=1000))]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        test_row = X[0].tolist()
        original_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
