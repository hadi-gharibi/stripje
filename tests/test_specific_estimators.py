#!/usr/bin/env python3
"""
Test script to verify various bug fixes and specific feature support.

This test ensures that specific estimators and transformers work correctly
with their optimized handlers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, LinearSVR

from stripje import compile_pipeline


class TestSpecificEstimators:
    """Test class for verifying specific estimator implementations."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        return X, y

    def test_gaussian_nb_support(self, classification_data):
        """Test that GaussianNB works correctly with optimized handler."""
        X, y = classification_data

        nb = GaussianNB()
        nb.fit(X, y)

        pipeline = Pipeline([("nb", nb)])
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)

        test_row = X[0].tolist()
        orig_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert (
            orig_pred == fast_pred
        ), f"GaussianNB predictions don't match: {orig_pred} vs {fast_pred}"

    def test_linear_svc_support(self, classification_data):
        """Test that LinearSVC is properly supported."""
        X, y = classification_data

        svc = LinearSVC(random_state=42, max_iter=1000)
        svc.fit(X, y)

        pipeline = Pipeline([("svc", svc)])
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)

        test_row = X[0].tolist()
        orig_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert (
            orig_pred == fast_pred
        ), f"LinearSVC predictions don't match: {orig_pred} vs {fast_pred}"

    def test_linear_svr_support(self, regression_data):
        """Test that LinearSVR is properly supported."""
        X, y = regression_data

        svr = LinearSVR(random_state=42, max_iter=1000)
        svr.fit(X, y)

        pipeline = Pipeline([("svr", svr)])
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)

        test_row = X[0].tolist()
        orig_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert np.allclose(
            [orig_pred], [fast_pred], rtol=1e-10
        ), f"LinearSVR predictions don't match: {orig_pred} vs {fast_pred}"

    def test_select_from_model_support(self, classification_data):
        """Test that SelectFromModel works properly."""
        X, y = classification_data

        estimator = LogisticRegression(random_state=42, max_iter=1000)
        selector = SelectFromModel(estimator)
        selector.fit(X, y)

        pipeline = Pipeline([("selector", selector)])

        fast_transform = compile_pipeline(pipeline)

        test_row = X[0].tolist()
        orig_result = selector.transform([test_row])[0]
        fast_result = fast_transform(test_row)

        assert (
            len(orig_result) == len(fast_result)
        ), f"SelectFromModel output shapes don't match: {len(orig_result)} vs {len(fast_result)}"
        assert np.allclose(
            orig_result, fast_result
        ), "SelectFromModel transformations don't match"

    def test_label_encoder_basic(self):
        """Test basic LabelEncoder functionality (not in pipeline context)."""
        labels = ["A", "B", "C", "A", "B"]
        le = LabelEncoder()
        le.fit(labels)

        # LabelEncoder is designed for target encoding, not feature transformation in pipelines
        assert list(le.classes_) == ["A", "B", "C"]
        assert le.transform(["A"])[0] == 0
        assert le.transform(["B"])[0] == 1
        assert le.transform(["C"])[0] == 2


if __name__ == "__main__":
    # Run tests manually if called directly
    print("Running specific estimator tests...")

    test_instance = TestSpecificEstimators()
    X_class, y_class = make_classification(
        n_samples=100, n_features=10, random_state=42
    )
    X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)

    try:
        test_instance.test_gaussian_nb_support((X_class, y_class))
        print("✓ GaussianNB support verified")
    except Exception as e:
        print(f"✗ GaussianNB test failed: {e}")

    try:
        test_instance.test_linear_svc_support((X_class, y_class))
        print("✓ LinearSVC support verified")
    except Exception as e:
        print(f"✗ LinearSVC test failed: {e}")

    try:
        test_instance.test_linear_svr_support((X_reg, y_reg))
        print("✓ LinearSVR support verified")
    except Exception as e:
        print(f"✗ LinearSVR test failed: {e}")

    try:
        test_instance.test_select_from_model_support((X_class, y_class))
        print("✓ SelectFromModel support verified")
    except Exception as e:
        print(f"✗ SelectFromModel test failed: {e}")

    try:
        test_instance.test_label_encoder_basic()
        print("✓ LabelEncoder basic functionality verified")
    except Exception as e:
        print(f"✗ LabelEncoder test failed: {e}")

    print("Specific estimator tests completed!")
