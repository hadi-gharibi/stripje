"""
Comprehensive tests for decomposition transformers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD

from stripje.transformers.decomposition import (
    handle_factor_analysis,
    handle_fast_ica,
    handle_pca,
    handle_truncated_svd,
)


class TestDecompositionTransformers:
    """Test suite for decomposition transformers."""

    @pytest.fixture
    def numeric_data(self):
        """Generate numeric test data."""
        X, _ = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_redundant=1,
            random_state=42,
        )
        return X

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, _ = make_regression(
            n_samples=100, n_features=10, n_informative=8, noise=0.1, random_state=42
        )
        return X

    @pytest.mark.parametrize("n_components", [3, 5, 8])
    def test_pca(self, numeric_data, n_components):
        """Test PCA handler."""
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(numeric_data)

        fast_pca = handle_pca(pca)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = pca.transform([test_row])[0]
            fast_result = fast_pca(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"PCA ({n_components} components) mismatch for row {i}"

    def test_pca_full_components(self, numeric_data):
        """Test PCA with all components."""
        pca = PCA(random_state=42)  # All components
        pca.fit(numeric_data)

        fast_pca = handle_pca(pca)

        test_row = numeric_data[0].tolist()
        original_result = pca.transform([test_row])[0]
        fast_result = fast_pca(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    @pytest.mark.parametrize("n_components", [3, 5, 8])
    def test_truncated_svd(self, numeric_data, n_components):
        """Test TruncatedSVD handler."""
        # Ensure data is non-negative for SVD
        X = np.abs(numeric_data)

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(X)

        fast_svd = handle_truncated_svd(svd)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = svd.transform([test_row])[0]
            fast_result = fast_svd(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"TruncatedSVD ({n_components} components) mismatch for row {i}"

    def test_truncated_svd_single_component(self, numeric_data):
        """Test TruncatedSVD with single component."""
        X = np.abs(numeric_data)

        svd = TruncatedSVD(n_components=1, random_state=42)
        svd.fit(X)

        fast_svd = handle_truncated_svd(svd)

        test_row = X[0].tolist()
        original_result = svd.transform([test_row])[0]
        fast_result = fast_svd(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    @pytest.mark.parametrize("n_components", [3, 5])
    def test_fast_ica(self, regression_data, n_components):
        """Test FastICA handler."""
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        ica.fit(regression_data)

        fast_ica = handle_fast_ica(ica)

        for i in range(5):
            test_row = regression_data[i].tolist()
            original_result = ica.transform([test_row])[0]
            fast_result = fast_ica(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"FastICA ({n_components} components) mismatch for row {i}"

    def test_fast_ica_whiten_false(self, regression_data):
        """Test FastICA with whiten=False."""
        ica = FastICA(n_components=3, whiten=False, random_state=42, max_iter=1000)
        ica.fit(regression_data)

        fast_ica = handle_fast_ica(ica)

        test_row = regression_data[0].tolist()
        original_result = ica.transform([test_row])[0]
        fast_result = fast_ica(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    @pytest.mark.parametrize("n_components", [3, 5])
    def test_factor_analysis(self, regression_data, n_components):
        """Test FactorAnalysis handler."""
        fa = FactorAnalysis(n_components=n_components, random_state=42)
        fa.fit(regression_data)

        fast_fa = handle_factor_analysis(fa)

        for i in range(5):
            test_row = regression_data[i].tolist()
            original_result = fa.transform([test_row])[0]
            fast_result = fast_fa(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"FactorAnalysis ({n_components} components) mismatch for row {i}"

    def test_factor_analysis_single_component(self, regression_data):
        """Test FactorAnalysis with single component."""
        fa = FactorAnalysis(n_components=1, random_state=42)
        fa.fit(regression_data)

        fast_fa = handle_factor_analysis(fa)

        test_row = regression_data[0].tolist()
        original_result = fa.transform([test_row])[0]
        fast_result = fast_fa(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    def test_decomposition_with_small_data(self):
        """Test decomposition transformers with minimal data."""
        # Create minimal dataset
        X = np.random.randn(10, 5)

        # Test PCA with small data
        pca = PCA(n_components=3)
        pca.fit(X)
        fast_pca = handle_pca(pca)

        test_row = X[0].tolist()
        original_result = pca.transform([test_row])[0]
        fast_result = fast_pca(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    def test_decomposition_edge_cases(self):
        """Test edge cases for decomposition transformers."""
        # Create data with perfect correlation
        X = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]])

        # Test PCA with correlated data
        pca = PCA(n_components=2)
        pca.fit(X)
        fast_pca = handle_pca(pca)

        test_row = X[0].tolist()
        original_result = pca.transform([test_row])[0]
        fast_result = fast_pca(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
