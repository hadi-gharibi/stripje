"""
Comprehensive tests for Naive Bayes estimator handlers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from stripje.estimators.naive_bayes import (
    handle_bernoulli_nb,
    handle_gaussian_nb,
    handle_multinomial_nb,
)


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
    def count_data(self):
        """Generate count-based data for MultinomialNB."""
        np.random.seed(42)
        X = np.random.randint(0, 10, size=(100, 5))
        y = np.random.randint(0, 2, size=100)
        return X, y

    @pytest.fixture
    def binary_feature_data(self):
        """Generate binary feature data for BernoulliNB."""
        np.random.seed(42)
        X = np.random.randint(0, 2, size=(100, 5))
        y = np.random.randint(0, 2, size=100)
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

            assert original_pred == fast_pred, (
                f"GaussianNB (binary) mismatch for row {i}: {original_pred} vs {fast_pred}"
            )

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

            assert original_pred == fast_pred, (
                f"GaussianNB (multiclass) mismatch for row {i}: {original_pred} vs {fast_pred}"
            )

    def test_multinomial_nb_binary(self, count_data):
        """Test MultinomialNB handler for binary classification."""
        X, y = count_data

        clf = MultinomialNB()
        clf.fit(X, y)

        fast_clf = handle_multinomial_nb(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"MultinomialNB (binary) mismatch for row {i}: {original_pred} vs {fast_pred}"
            )

    def test_multinomial_nb_multiclass(self, count_data):
        """Test MultinomialNB handler for multiclass classification."""
        X, y = count_data
        # Convert to multiclass
        y = np.random.randint(0, 3, size=len(y))

        clf = MultinomialNB()
        clf.fit(X, y)

        fast_clf = handle_multinomial_nb(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"MultinomialNB (multiclass) mismatch for row {i}: {original_pred} vs {fast_pred}"
            )

    def test_bernoulli_nb_binary(self, binary_feature_data):
        """Test BernoulliNB handler for binary classification."""
        X, y = binary_feature_data

        clf = BernoulliNB()
        clf.fit(X, y)

        fast_clf = handle_bernoulli_nb(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"BernoulliNB (binary) mismatch for row {i}: {original_pred} vs {fast_pred}"
            )

    def test_bernoulli_nb_multiclass(self, binary_feature_data):
        """Test BernoulliNB handler for multiclass classification."""
        X, y = binary_feature_data
        # Convert to multiclass
        y = np.random.randint(0, 3, size=len(y))

        clf = BernoulliNB()
        clf.fit(X, y)

        fast_clf = handle_bernoulli_nb(clf)

        for i in range(10):
            test_row = X[i].tolist()
            original_pred = clf.predict([test_row])[0]
            fast_pred = fast_clf(test_row)

            assert original_pred == fast_pred, (
                f"BernoulliNB (multiclass) mismatch for row {i}: {original_pred} vs {fast_pred}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
