"""
Comprehensive tests for tree-based and other estimators.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stripje.estimators.tree import (
    handle_decision_tree_classifier,
    handle_decision_tree_regressor,
)


class TestTreeEstimators:
    """Test suite for tree-based estimators."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=100, n_features=5, n_informative=3, noise=0.1, random_state=42
        )
        return X, y

    @pytest.mark.parametrize("max_depth", [None, 3, 5, 10])
    def test_decision_tree_classifier(self, classification_data, max_depth):
        """Test DecisionTreeClassifier handler."""
        X, y = classification_data

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X, y)

        fast_clf = handle_decision_tree_classifier(clf)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"DecisionTreeClassifier (max_depth={max_depth}) mismatch for row {i}"

    @pytest.mark.parametrize("max_depth", [None, 3, 5, 10])
    def test_decision_tree_regressor(self, regression_data, max_depth):
        """Test DecisionTreeRegressor handler."""
        X, y = regression_data

        reg = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        reg.fit(X, y)

        fast_reg = handle_decision_tree_regressor(reg)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"DecisionTreeRegressor (max_depth={max_depth}) mismatch for row {i}"

    def test_decision_tree_classifier_binary(self):
        """Test DecisionTreeClassifier with binary classification."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_classes=2,
            n_informative=2,
            n_redundant=1,
            random_state=42,
        )

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)

        fast_clf = handle_decision_tree_classifier(clf)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred

    def test_decision_tree_edge_cases(self):
        """Test edge cases for decision trees."""
        # Single feature
        X = np.random.randn(50, 1)
        y = (X[:, 0] > 0).astype(int)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)

        fast_clf = handle_decision_tree_classifier(clf)

        test_row = [0.5]
        original_pred = clf.predict([test_row])[0]
        fast_pred = fast_clf(test_row)

        assert original_pred == fast_pred

        # Perfect separation
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)

        fast_clf = handle_decision_tree_classifier(clf)

        for i in range(len(X)):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred

    def test_decision_tree_different_criteria(self, classification_data):
        """Test DecisionTreeClassifier with different splitting criteria."""
        X, y = classification_data

        for criterion in ["gini", "entropy"]:
            clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
            clf.fit(X, y)

            fast_clf = handle_decision_tree_classifier(clf)

            test_row = X[0].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, f"Mismatch with criterion {criterion}"

    def test_decision_tree_min_samples_split(self, classification_data):
        """Test DecisionTreeClassifier with min_samples_split parameter."""
        X, y = classification_data

        for min_samples in [2, 5, 10]:
            clf = DecisionTreeClassifier(min_samples_split=min_samples, random_state=42)
            clf.fit(X, y)

            fast_clf = handle_decision_tree_classifier(clf)

            test_row = X[0].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"Mismatch with min_samples_split {min_samples}"

    def test_tree_traversal_consistency(self, classification_data):
        """Test that tree traversal is consistent across multiple calls."""
        X, y = classification_data

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)

        fast_clf = handle_decision_tree_classifier(clf)

        test_row = X[0].tolist()

        # Call multiple times and ensure consistency
        predictions = [fast_clf(test_row) for _ in range(10)]

        # All predictions should be identical
        for pred in predictions[1:]:
            assert pred == predictions[0]

    def test_tree_with_feature_importance(self, classification_data):
        """Test that trees with different feature importances work correctly."""
        X, y = classification_data

        # Create tree with specific parameters that might affect feature importance
        clf = DecisionTreeClassifier(
            max_features="sqrt", min_samples_leaf=5, random_state=42
        )
        clf.fit(X, y)

        fast_clf = handle_decision_tree_classifier(clf)

        # Test multiple samples to ensure robustness
        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
