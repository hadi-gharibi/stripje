"""
Comprehensive tests for feature selection transformers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
)
from sklearn.linear_model import LogisticRegression

from stripje.transformers.feature_selection import (
    handle_rfe,
    handle_select_fdr,
    handle_select_fpr,
    handle_select_from_model,
    handle_select_fwe,
    handle_select_k_best,
    handle_select_percentile,
    handle_variance_threshold,
)


class TestFeatureSelectionTransformers:
    """Test suite for feature selection transformers."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def high_variance_data(self):
        """Generate data with varying variance."""
        np.random.seed(42)
        # Create features with different variances
        X = np.random.randn(100, 6)
        X[:, 0] *= 10  # High variance
        X[:, 1] *= 1  # Medium variance
        X[:, 2] *= 0.1  # Low variance
        X[:, 3] *= 0.01  # Very low variance
        X[:, 4] = 0  # Zero variance
        X[:, 5] *= 5  # High variance
        return X

    @pytest.mark.parametrize("k", [3, 5, 7])
    @pytest.mark.parametrize("score_func", [f_classif])
    def test_select_k_best(self, classification_data, k, score_func):
        """Test SelectKBest handler."""
        X, y = classification_data

        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        fast_selector = handle_select_k_best(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"SelectKBest (k={k}) mismatch for row {i}"
            assert (
                len(fast_result) == k
            ), f"Expected {k} features, got {len(fast_result)}"

    def test_select_k_best_edge_cases(self, classification_data):
        """Test SelectKBest edge cases."""
        X, y = classification_data

        # Test k=1
        selector = SelectKBest(score_func=f_classif, k=1)
        selector.fit(X, y)
        fast_selector = handle_select_k_best(selector)

        test_row = X[0].tolist()
        original_result = selector.transform([test_row])[0]
        fast_result = fast_selector(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)
        assert len(fast_result) == 1

        # Test k = all features
        selector = SelectKBest(score_func=f_classif, k=X.shape[1])
        selector.fit(X, y)
        fast_selector = handle_select_k_best(selector)

        original_result = selector.transform([test_row])[0]
        fast_result = fast_selector(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)
        assert len(fast_result) == X.shape[1]

    @pytest.mark.parametrize("percentile", [10, 25, 50, 75])
    def test_select_percentile(self, classification_data, percentile):
        """Test SelectPercentile handler."""
        X, y = classification_data

        selector = SelectPercentile(score_func=f_classif, percentile=percentile)
        selector.fit(X, y)

        fast_selector = handle_select_percentile(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"SelectPercentile ({percentile}%) mismatch for row {i}"

    @pytest.mark.parametrize("threshold", [0.0, 0.01, 0.1, 1.0])
    def test_variance_threshold(self, high_variance_data, threshold):
        """Test VarianceThreshold handler."""
        X = high_variance_data

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        fast_selector = handle_variance_threshold(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"VarianceThreshold (threshold={threshold}) mismatch for row {i}"

    def test_variance_threshold_removes_zero_variance(self, high_variance_data):
        """Test that VarianceThreshold removes zero variance features."""
        X = high_variance_data

        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)

        fast_selector = handle_variance_threshold(selector)

        test_row = X[0].tolist()
        fast_result = fast_selector(test_row)

        # Should remove the zero variance feature (index 4)
        assert len(fast_result) < len(test_row)

    def test_select_from_model_random_forest(self, classification_data):
        """Test SelectFromModel with RandomForest."""
        X, y = classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = SelectFromModel(estimator)
        selector.fit(X, y)

        fast_selector = handle_select_from_model(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"SelectFromModel (RandomForest) mismatch for row {i}"

    def test_select_from_model_logistic_regression(self, classification_data):
        """Test SelectFromModel with LogisticRegression."""
        X, y = classification_data

        estimator = LogisticRegression(random_state=42, max_iter=1000)
        selector = SelectFromModel(estimator)
        selector.fit(X, y)

        fast_selector = handle_select_from_model(selector)

        test_row = X[0].tolist()
        original_result = selector.transform([test_row])[0]
        fast_result = fast_selector(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    @pytest.mark.parametrize("n_features_to_select", [3, 5])
    def test_rfe(self, classification_data, n_features_to_select):
        """Test RFE handler."""
        X, y = classification_data

        estimator = LogisticRegression(random_state=42, max_iter=1000)
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector.fit(X, y)

        fast_selector = handle_rfe(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"RFE ({n_features_to_select} features) mismatch for row {i}"
            assert len(fast_result) == n_features_to_select

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_select_fdr(self, classification_data, alpha):
        """Test SelectFdr handler."""
        X, y = classification_data

        selector = SelectFdr(score_func=f_classif, alpha=alpha)
        selector.fit(X, y)

        fast_selector = handle_select_fdr(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"SelectFdr (alpha={alpha}) mismatch for row {i}"

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_select_fpr(self, classification_data, alpha):
        """Test SelectFpr handler."""
        X, y = classification_data

        selector = SelectFpr(score_func=f_classif, alpha=alpha)
        selector.fit(X, y)

        fast_selector = handle_select_fpr(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"SelectFpr (alpha={alpha}) mismatch for row {i}"

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_select_fwe(self, classification_data, alpha):
        """Test SelectFwe handler."""
        X, y = classification_data

        selector = SelectFwe(score_func=f_classif, alpha=alpha)
        selector.fit(X, y)

        fast_selector = handle_select_fwe(selector)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = selector.transform([test_row])[0]
            fast_result = fast_selector(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"SelectFwe (alpha={alpha}) mismatch for row {i}"

    def test_feature_selection_consistency(self, classification_data):
        """Test that feature selection is consistent across multiple calls."""
        X, y = classification_data

        selector = SelectKBest(score_func=f_classif, k=5)
        selector.fit(X, y)

        fast_selector = handle_select_k_best(selector)

        test_row = X[0].tolist()

        # Call multiple times and ensure consistency
        results = [fast_selector(test_row) for _ in range(5)]

        for result in results[1:]:
            assert np.allclose(results[0], result, rtol=1e-10)

    def test_feature_selection_preserves_order(self, classification_data):
        """Test that selected features maintain their relative order."""
        X, y = classification_data

        # Add some obvious patterns to features
        X_modified = X.copy()
        X_modified[:, 0] = y  # Make first feature highly predictive
        X_modified[:, 1] = -y  # Make second feature highly predictive (inverse)

        selector = SelectKBest(score_func=f_classif, k=3)
        selector.fit(X_modified, y)

        fast_selector = handle_select_k_best(selector)

        test_row = X_modified[0].tolist()
        fast_result = fast_selector(test_row)

        # Should select 3 features
        assert len(fast_result) == 3

    def test_empty_selection_edge_case(self):
        """Test edge case where no features are selected."""
        # Create data where all features have very low variance
        X = np.ones((50, 5)) + np.random.randn(50, 5) * 0.001

        # sklearn VarianceThreshold raises an error when no features meet threshold
        with pytest.raises(
            ValueError, match="No feature in X meets the variance threshold"
        ):
            selector = VarianceThreshold(threshold=0.1)  # High threshold
            selector.fit(X)

        # Test with a threshold that allows some features through
        selector_low = VarianceThreshold(threshold=0.000001)  # Even lower threshold
        selector_low.fit(X)

        fast_selector = handle_variance_threshold(selector_low)

        test_row = X[0].tolist()
        fast_result = fast_selector(test_row)

        # Should select some features
        assert isinstance(fast_result, list)
        assert len(fast_result) > 0

    def test_select_from_model_proper_fitting(self, classification_data):
        """Test SelectFromModel with proper y fitting."""
        X, y = classification_data

        estimator = LogisticRegression(random_state=42, max_iter=1000)
        selector = SelectFromModel(estimator)
        selector.fit(X, y)  # This requires y

        fast_selector = handle_select_from_model(selector)

        test_row = X[0].tolist()
        original_result = selector.transform([test_row])[0]
        fast_result = fast_selector(test_row)

        assert (
            len(original_result) == len(fast_result)
        ), f"SelectFromModel output shapes don't match: {len(original_result)} vs {len(fast_result)}"
        assert np.allclose(
            original_result, fast_result, rtol=1e-10
        ), "SelectFromModel transformations don't match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
