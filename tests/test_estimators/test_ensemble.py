"""
Comprehensive tests for ensemble estimator handlers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from stripje.estimators.ensemble import (
    handle_gradient_boosting_classifier,
    handle_gradient_boosting_regressor,
    handle_random_forest_classifier,
    handle_random_forest_regressor,
)


class TestEnsembleEstimators:
    """Test suite for ensemble model estimators."""

    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def multiclass_classification_data(self):
        """Generate multiclass classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=5,
            n_redundant=2,
            n_classes=4,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=200, n_features=8, n_informative=5, noise=0.1, random_state=42
        )
        return X, y

    # Random Forest Tests
    def test_random_forest_classifier_binary(self, binary_classification_data):
        """Test RandomForestClassifier handler for binary classification."""
        X, y = binary_classification_data

        clf = RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=42, min_samples_split=5
        )
        clf.fit(X, y)

        fast_clf = handle_random_forest_classifier(clf)

        # Test multiple rows to ensure consistency
        for i in range(20):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"RandomForestClassifier (binary) mismatch for row {i}: "
                f"{original_pred} vs {fast_pred}"
            )

    def test_random_forest_classifier_multiclass(self, multiclass_classification_data):
        """Test RandomForestClassifier handler for multiclass classification."""
        X, y = multiclass_classification_data

        clf = RandomForestClassifier(
            n_estimators=15, max_depth=4, random_state=42, min_samples_split=5
        )
        clf.fit(X, y)

        fast_clf = handle_random_forest_classifier(clf)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"RandomForestClassifier (multiclass) mismatch for row {i}: "
                f"{original_pred} vs {fast_pred}"
            )

    def test_random_forest_classifier_edge_cases(self, binary_classification_data):
        """Test RandomForestClassifier with different configurations."""
        X, y = binary_classification_data

        # Test with different numbers of estimators
        for n_estimators in [1, 5, 10]:
            clf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=3, random_state=42
            )
            clf.fit(X, y)
            fast_clf = handle_random_forest_classifier(clf)

            test_row = X[0].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"RandomForestClassifier with {n_estimators} estimators failed"

    def test_random_forest_regressor(self, regression_data):
        """Test RandomForestRegressor handler."""
        X, y = regression_data

        reg = RandomForestRegressor(
            n_estimators=20, max_depth=5, random_state=42, min_samples_split=5
        )
        reg.fit(X, y)

        fast_reg = handle_random_forest_regressor(reg)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            # Use relative tolerance for regression predictions
            assert np.isclose(original_pred, fast_pred, rtol=1e-10), (
                f"RandomForestRegressor mismatch for row {i}: "
                f"{original_pred} vs {fast_pred}"
            )

    def test_random_forest_regressor_edge_cases(self, regression_data):
        """Test RandomForestRegressor with different configurations."""
        X, y = regression_data

        # Test with single estimator
        reg = RandomForestRegressor(n_estimators=1, max_depth=3, random_state=42)
        reg.fit(X, y)
        fast_reg = handle_random_forest_regressor(reg)

        test_row = X[0].tolist()
        original_pred = reg.predict([test_row])[0]
        fast_pred = fast_reg(test_row)

        assert np.isclose(original_pred, fast_pred, rtol=1e-10)

    # Gradient Boosting Tests
    def test_gradient_boosting_classifier_binary(self, binary_classification_data):
        """Test GradientBoostingClassifier handler for binary classification."""
        X, y = binary_classification_data

        clf = GradientBoostingClassifier(
            n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42
        )
        clf.fit(X, y)

        fast_clf = handle_gradient_boosting_classifier(clf)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"GradientBoostingClassifier (binary) mismatch for row {i}: "
                f"{original_pred} vs {fast_pred}"
            )

    def test_gradient_boosting_classifier_multiclass(
        self, multiclass_classification_data
    ):
        """Test GradientBoostingClassifier handler for multiclass classification."""
        X, y = multiclass_classification_data

        clf = GradientBoostingClassifier(
            n_estimators=15, learning_rate=0.1, max_depth=3, random_state=42
        )
        clf.fit(X, y)

        fast_clf = handle_gradient_boosting_classifier(clf)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"GradientBoostingClassifier (multiclass) mismatch for row {i}: "
                f"{original_pred} vs {fast_pred}"
            )

    def test_gradient_boosting_classifier_probability_threshold(
        self, binary_classification_data
    ):
        """Test GradientBoostingClassifier probability threshold behavior."""
        X, y = binary_classification_data

        clf = GradientBoostingClassifier(
            n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42
        )
        clf.fit(X, y)

        fast_clf = handle_gradient_boosting_classifier(clf)

        # Test edge cases around 0.5 probability
        for i in range(len(X)):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            original_proba = clf.predict_proba([test_row])[0]
            fast_pred = fast_clf(test_row)

            # Verify prediction matches probability threshold
            expected_pred = (
                clf.classes_[1] if original_proba[1] > 0.5 else clf.classes_[0]
            )
            assert original_pred == expected_pred == fast_pred

    def test_gradient_boosting_classifier_learning_rates(
        self, binary_classification_data
    ):
        """Test GradientBoostingClassifier with different learning rates."""
        X, y = binary_classification_data

        for learning_rate in [0.01, 0.1, 0.2]:
            clf = GradientBoostingClassifier(
                n_estimators=10,
                learning_rate=learning_rate,
                max_depth=3,
                random_state=42,
            )
            clf.fit(X, y)
            fast_clf = handle_gradient_boosting_classifier(clf)

            test_row = X[0].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"GradientBoostingClassifier with learning_rate={learning_rate} failed"

    def test_gradient_boosting_regressor(self, regression_data):
        """Test GradientBoostingRegressor handler."""
        X, y = regression_data

        reg = GradientBoostingRegressor(
            n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42
        )
        reg.fit(X, y)

        fast_reg = handle_gradient_boosting_regressor(reg)

        for i in range(20):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(original_pred, fast_pred, rtol=1e-10), (
                f"GradientBoostingRegressor mismatch for row {i}: "
                f"{original_pred} vs {fast_pred}"
            )

    def test_gradient_boosting_regressor_learning_rates(self, regression_data):
        """Test GradientBoostingRegressor with different learning rates."""
        X, y = regression_data

        for learning_rate in [0.01, 0.1, 0.2]:
            reg = GradientBoostingRegressor(
                n_estimators=10,
                learning_rate=learning_rate,
                max_depth=3,
                random_state=42,
            )
            reg.fit(X, y)
            fast_reg = handle_gradient_boosting_regressor(reg)

            test_row = X[0].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"GradientBoostingRegressor with learning_rate={learning_rate} failed"

    # Edge Case and Performance Tests
    def test_ensemble_models_with_small_datasets(self):
        """Test ensemble models with minimal datasets."""
        # Small binary classification dataset
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])

        clf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
        clf.fit(X, y)
        fast_clf = handle_random_forest_classifier(clf)

        test_row = [2.5, 3.5]
        original_pred = clf.predict([test_row])[0]
        fast_pred = fast_clf(test_row)

        assert original_pred == fast_pred

        # Small regression dataset
        X_reg = np.array([[1], [2], [3], [4], [5]])
        y_reg = np.array([2, 4, 6, 8, 10])

        reg = GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42)
        reg.fit(X_reg, y_reg)
        fast_reg = handle_gradient_boosting_regressor(reg)

        test_row = [3.5]
        original_pred = reg.predict([test_row])[0]
        fast_pred = fast_reg(test_row)

        assert np.isclose(original_pred, fast_pred, rtol=1e-10)

    def test_ensemble_models_with_single_feature(self, binary_classification_data):
        """Test ensemble models with single feature."""
        X, y = binary_classification_data
        X_single = X[:, :1]  # Use only first feature

        clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X_single, y)
        fast_clf = handle_random_forest_classifier(clf)

        test_row = [X_single[0, 0]]
        original_pred = clf.predict([test_row])[0]
        fast_pred = fast_clf(test_row)

        assert original_pred == fast_pred

    def test_ensemble_models_class_ordering(self):
        """Test that ensemble models handle different class orderings correctly."""
        # Create dataset with non-standard class labels
        X = np.random.randn(100, 5)
        y = np.random.choice([5, 10, 15], size=100)  # Non-standard class labels

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        fast_clf = handle_random_forest_classifier(clf)

        # Verify classes are handled correctly
        assert np.array_equal(clf.classes_, [5, 10, 15])

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred
            assert fast_pred in clf.classes_

    def test_gradient_boosting_extreme_cases(self):
        """Test gradient boosting with extreme cases."""
        # Perfect separable data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])  # XOR pattern

        clf = GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
        )
        clf.fit(X, y)
        fast_clf = handle_gradient_boosting_classifier(clf)

        for i in range(len(X)):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred

    def test_ensemble_deep_trees(self, binary_classification_data):
        """Test ensemble models with deeper trees."""
        X, y = binary_classification_data

        # Test with deeper trees
        clf = RandomForestClassifier(
            n_estimators=5, max_depth=10, min_samples_split=2, random_state=42
        )
        clf.fit(X, y)
        fast_clf = handle_random_forest_classifier(clf)

        test_row = X[0].tolist()
        original_pred = clf.predict([test_row])[0]
        fast_pred = fast_clf(test_row)

        assert original_pred == fast_pred

    def test_ensemble_models_consistency_across_predictions(self, regression_data):
        """Test that ensemble models produce consistent predictions across multiple calls."""
        X, y = regression_data

        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X, y)
        fast_reg = handle_random_forest_regressor(reg)

        test_row = X[0].tolist()

        # Test multiple calls to ensure consistency
        predictions = [fast_reg(test_row) for _ in range(5)]

        # All predictions should be identical
        assert all(
            np.isclose(p, predictions[0], rtol=1e-15) for p in predictions
        ), "RandomForestRegressor should produce consistent predictions"

        # Should match original prediction
        original_pred = reg.predict([test_row])[0]
        assert np.isclose(original_pred, predictions[0], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
