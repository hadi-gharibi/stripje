"""
Comprehensive tests for Naive Bayes estimator handlers.
"""

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.naive_bayes import GaussianNB

from stripje.estimators.naive_bayes import handle_gaussian_nb


class TestNaiveBayesEstimators:
    """Test suite for Naive Bayes estimators handlers."""

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

    def test_gaussian_nb_binary(self, binary_classification_data):
        """Test GaussianNB handler for binary classification."""
        X, y = binary_classification_data

        clf = GaussianNB()
        clf.fit(X, y)

        fast_clf = handle_gaussian_nb(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"GaussianNB (binary) mismatch for row {i}: {original_pred} vs {fast_pred}"

    def test_gaussian_nb_multiclass(self, multiclass_classification_data):
        """Test GaussianNB handler for multiclass classification."""
        X, y = multiclass_classification_data

        clf = GaussianNB()
        clf.fit(X, y)

        fast_clf = handle_gaussian_nb(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert (
                original_pred == fast_pred
            ), f"GaussianNB (multiclass) mismatch for row {i}: {original_pred} vs {fast_pred}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
