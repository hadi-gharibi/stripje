#!/usr/bin/env python3
"""
Test script to verify the fallback mechanism for unsupported transformers and estimators.

This test ensures that when a transformer or estimator doesn't have an optimized handler,
the system falls back to using the original implementation instead of failing.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from stripje import compile_pipeline


class CustomTransformer(BaseEstimator, TransformerMixin):
    """A custom transformer that won't have a handler."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Simple transformation: multiply by 2
        return X * 2


class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    """A custom transformer for use in ColumnTransformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Simple transformation: add 1 to all values
        return X + 1


class TestFallbackMechanism:
    """Test class for verifying the fallback mechanism."""

    @pytest.fixture
    def test_data(self):
        """Generate test data."""
        X, y = make_classification(n_samples=20, n_features=4, random_state=42)
        return X, y

    def test_custom_transformer_fallback(self, test_data):
        """Test fallback mechanism with custom transformer."""
        """Test that custom transformers fall back to original implementation."""
        X, y = test_data

        custom_transformer = CustomTransformer()
        custom_transformer.fit(X)

        pipeline = Pipeline(
            [
                ("custom", custom_transformer),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        pipeline.fit(X, y)

        # This should work with fallback mechanism
        fast_predict = compile_pipeline(pipeline)

        test_row = X[0].tolist()
        orig_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert orig_pred == fast_pred, (
            f"Custom transformer fallback failed: {orig_pred} vs {fast_pred}"
        )

    def test_target_encoder_fallback(self):
        """Test that TargetEncoder falls back to original implementation."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )

        # Convert to categorical features for TargetEncoder
        X_cat = X.astype(str)

        target_encoder = TargetEncoder(random_state=42)
        target_encoder.fit(X_cat, y)

        pipeline = Pipeline(
            [
                ("target_encoder", target_encoder),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        pipeline.fit(X_cat, y)

        fast_predict = compile_pipeline(pipeline)
        test_row = X_cat[0].tolist()

        orig_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert orig_pred == fast_pred, (
            f"TargetEncoder fallback failed: {orig_pred} vs {fast_pred}"
        )

    def test_column_transformer_fallback(self, test_data):
        """Test that fallback works within ColumnTransformer."""
        X, y = test_data
        X = X[:, :4]  # Use only 4 features for this test

        # Create ColumnTransformer with mix of supported and unsupported transformers
        column_transformer = ColumnTransformer(
            [
                ("standard", StandardScaler(), [0, 1]),  # Should have handler
                ("custom", CustomColumnTransformer(), [2, 3]),  # Should use fallback
            ]
        )

        pipeline = Pipeline(
            [
                ("column_trans", column_transformer),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)
        test_row = X[0].tolist()

        orig_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert orig_pred == fast_pred, (
            f"ColumnTransformer fallback failed: {orig_pred} vs {fast_pred}"
        )

    def test_fallback_single_value_output(self):
        """Test fallback with transformer that outputs single value."""
        from sklearn.feature_selection import VarianceThreshold

        X = np.array([[1, 2, 0], [4, 5, 0], [7, 8, 0], [2, 3, 0]])
        y = np.array([0, 1, 0, 1])

        # VarianceThreshold will remove the zero-variance column
        selector = VarianceThreshold()
        selector.fit(X)

        # Create pipeline with unsupported transformer (for testing)
        class SingleOutputTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # Return single column
                return X[:, [0]]

        transformer = SingleOutputTransformer()
        transformer.fit(X)

        from stripje.registry import create_fallback_handler

        fallback = create_fallback_handler(transformer)
        test_row = [1, 2, 3]
        result = fallback(test_row)

        # Should return single value
        assert isinstance(result, (int, float, np.number))

    def test_fallback_invalid_step(self):
        """Test fallback with object that has neither transform nor predict."""

        class InvalidStep:
            pass

        invalid_step = InvalidStep()

        from stripje.registry import create_fallback_handler

        fallback = create_fallback_handler(invalid_step)

        with pytest.raises(ValueError, match="neither transform nor predict"):
            fallback([1, 2, 3])


if __name__ == "__main__":
    # Run tests manually if called directly
    print("Running fallback mechanism tests...")

    test_instance = TestFallbackMechanism()
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    try:
        test_instance.test_custom_transformer_fallback((X, y))
        print("✓ Custom transformer fallback verified")
    except Exception as e:
        print(f"✗ Custom transformer fallback failed: {e}")

    try:
        test_instance.test_target_encoder_fallback()
        print("✓ TargetEncoder fallback verified")
    except Exception as e:
        print(f"✗ TargetEncoder fallback failed: {e}")

    try:
        test_instance.test_column_transformer_fallback((X, y))
        print("✓ ColumnTransformer fallback verified")
    except Exception as e:
        print(f"✗ ColumnTransformer fallback failed: {e}")

    print("Fallback mechanism tests completed!")
