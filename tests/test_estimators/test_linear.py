"""
Comprehensive tests for linear estimator handlers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)

from stripje.estimators.linear import (
    handle_elastic_net,
    handle_lasso,
    handle_linear_regression,
    handle_logistic_regression,
    handle_ridge,
    handle_ridge_classifier,
    handle_sgd_classifier,
    handle_sgd_regressor,
)


class TestLinearEstimators:
    """Test suite for linear model estimators."""

    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def multiclass_classification_data(self):
        """Generate multiclass classification test data."""
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

    def test_logistic_regression_binary(self, binary_classification_data):
        """Test LogisticRegression handler for binary classification."""
        X, y = binary_classification_data

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X, y)

        fast_clf = handle_logistic_regression(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"LogisticRegression (binary) mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_logistic_regression_multiclass(self, multiclass_classification_data):
        """Test LogisticRegression handler for multiclass classification."""
        X, y = multiclass_classification_data

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X, y)

        fast_clf = handle_logistic_regression(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"LogisticRegression (multiclass) mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_logistic_regression_probability_threshold(
        self, binary_classification_data
    ):
        """Test LogisticRegression probability threshold behavior."""
        X, y = binary_classification_data

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X, y)

        fast_clf = handle_logistic_regression(clf)

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

    def test_linear_regression(self, regression_data):
        """Test LinearRegression handler."""
        X, y = regression_data

        reg = LinearRegression()
        reg.fit(X, y)

        fast_reg = handle_linear_regression(reg)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"LinearRegression mismatch for row {i}: {original_pred} vs {fast_pred}"

    @pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
    def test_ridge_regression(self, regression_data, alpha):
        """Test Ridge regression handler."""
        X, y = regression_data

        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(X, y)

        fast_reg = handle_ridge(reg)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"Ridge (alpha={alpha}) mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_ridge_classifier_binary(self, binary_classification_data):
        """Test RidgeClassifier handler for binary classification."""
        X, y = binary_classification_data

        clf = RidgeClassifier(random_state=42)
        clf.fit(X, y)

        fast_clf = handle_ridge_classifier(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"RidgeClassifier (binary) mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_ridge_classifier_multiclass(self, multiclass_classification_data):
        """Test RidgeClassifier handler for multiclass classification."""
        X, y = multiclass_classification_data

        clf = RidgeClassifier(random_state=42)
        clf.fit(X, y)

        fast_clf = handle_ridge_classifier(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"RidgeClassifier (multiclass) mismatch for row {i}: {original_pred} vs {fast_pred}"

    @pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
    def test_lasso_regression(self, regression_data, alpha):
        """Test Lasso regression handler."""
        X, y = regression_data

        reg = Lasso(alpha=alpha, random_state=42, max_iter=1000)
        reg.fit(X, y)

        fast_reg = handle_lasso(reg)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"Lasso (alpha={alpha}) mismatch for row {i}: {original_pred} vs {fast_pred}"

    @pytest.mark.parametrize("alpha", [0.1, 1.0])
    @pytest.mark.parametrize("l1_ratio", [0.15, 0.5, 0.85])
    def test_elastic_net_regression(self, regression_data, alpha, l1_ratio):
        """Test ElasticNet regression handler."""
        X, y = regression_data

        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=1000)
        reg.fit(X, y)

        fast_reg = handle_elastic_net(reg)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio}) mismatch for row {i}"

    def test_sgd_classifier_binary(self, binary_classification_data):
        """Test SGDClassifier handler for binary classification."""
        X, y = binary_classification_data

        clf = SGDClassifier(random_state=42, max_iter=1000)
        clf.fit(X, y)

        fast_clf = handle_sgd_classifier(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"SGDClassifier (binary) mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_sgd_classifier_multiclass(self, multiclass_classification_data):
        """Test SGDClassifier handler for multiclass classification."""
        X, y = multiclass_classification_data

        clf = SGDClassifier(random_state=42, max_iter=1000)
        clf.fit(X, y)

        fast_clf = handle_sgd_classifier(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"SGDClassifier (multiclass) mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_sgd_regressor(self, regression_data):
        """Test SGDRegressor handler."""
        X, y = regression_data

        reg = SGDRegressor(random_state=42, max_iter=1000)
        reg.fit(X, y)

        fast_reg = handle_sgd_regressor(reg)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = reg.predict([test_row])[0]
            fast_pred = fast_reg(test_row)

            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"SGDRegressor mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_linear_models_with_intercept_false(self, regression_data):
        """Test linear models with fit_intercept=False."""
        X, y = regression_data

        # Test LinearRegression without intercept
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        fast_reg = handle_linear_regression(reg)

        test_row = X[0].tolist()
        original_pred = reg.predict([test_row])[0]
        fast_pred = fast_reg(test_row)

        assert np.isclose(original_pred, fast_pred, rtol=1e-10)

        # Test Ridge without intercept
        reg = Ridge(fit_intercept=False, random_state=42)
        reg.fit(X, y)
        fast_reg = handle_ridge(reg)

        original_pred = reg.predict([test_row])[0]
        fast_pred = fast_reg(test_row)

        assert np.isclose(original_pred, fast_pred, rtol=1e-10)

    def test_linear_models_edge_cases(self):
        """Test linear models with edge case data."""
        # Single feature
        X = np.random.randn(50, 1)
        y = X[:, 0] * 2 + 1 + np.random.randn(50) * 0.1

        reg = LinearRegression()
        reg.fit(X, y)
        fast_reg = handle_linear_regression(reg)

        test_row = [1.5]
        original_pred = reg.predict([test_row])[0]
        fast_pred = fast_reg(test_row)

        assert np.isclose(original_pred, fast_pred, rtol=1e-10)

        # Perfect linear relationship
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        reg = LinearRegression()
        reg.fit(X, y)
        fast_reg = handle_linear_regression(reg)

        test_row = [3]
        original_pred = reg.predict([test_row])[0]
        fast_pred = fast_reg(test_row)

        assert np.isclose(original_pred, fast_pred, rtol=1e-10)

    def test_regularized_models_sparsity(self, regression_data):
        """Test that regularized models handle sparsity correctly."""
        X, y = regression_data

        # Use high regularization to create sparse coefficients
        reg = Lasso(alpha=10.0, random_state=42, max_iter=1000)
        reg.fit(X, y)

        fast_reg = handle_lasso(reg)

        # Check that zero coefficients are handled correctly
        test_row = X[0].tolist()
        original_pred = reg.predict([test_row])[0]
        fast_pred = fast_reg(test_row)

        assert np.isclose(original_pred, fast_pred, rtol=1e-10)

        # Verify some coefficients are actually zero
        assert (
            np.sum(np.abs(reg.coef_) < 1e-10) > 0
        ), "Lasso should produce some zero coefficients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
