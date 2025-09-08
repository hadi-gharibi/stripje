"""
Comprehensive tests for preprocessing transformers.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from stripje.transformers.preprocessing import (
    handle_binarizer,
    handle_kbins_discretizer,
    handle_label_binarizer,
    handle_label_encoder,
    handle_maxabs_scaler,
    handle_minmax_scaler,
    handle_normalizer,
    handle_onehot_encoder,
    handle_ordinal_encoder,
    handle_power_transformer,
    handle_quantile_transformer,
    handle_robust_scaler,
    handle_standard_scaler,
    handle_target_encoder,
)


class TestPreprocessingTransformers:
    """Test suite for preprocessing transformers."""

    @pytest.fixture
    def numeric_data(self):
        """Generate numeric test data."""
        X, _ = make_classification(
            n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42
        )
        return X

    @pytest.fixture
    def categorical_data(self):
        """Generate categorical test data."""
        return np.array([["A", "X"], ["B", "Y"], ["C", "Z"], ["A", "X"], ["B", "Y"]])

    @pytest.fixture
    def label_data(self):
        """Generate label data."""
        return np.array(["cat", "dog", "bird", "cat", "dog"])

    def test_standard_scaler(self, numeric_data):
        """Test StandardScaler handler."""
        scaler = StandardScaler()
        scaler.fit(numeric_data)

        fast_scaler = handle_standard_scaler(scaler)

        # Test multiple rows
        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = scaler.transform([test_row])[0]
            fast_result = fast_scaler(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"StandardScaler mismatch for row {i}"

    def test_minmax_scaler(self, numeric_data):
        """Test MinMaxScaler handler."""
        scaler = MinMaxScaler()
        scaler.fit(numeric_data)

        fast_scaler = handle_minmax_scaler(scaler)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = scaler.transform([test_row])[0]
            fast_result = fast_scaler(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"MinMaxScaler mismatch for row {i}"

    def test_robust_scaler(self, numeric_data):
        """Test RobustScaler handler."""
        scaler = RobustScaler()
        scaler.fit(numeric_data)

        fast_scaler = handle_robust_scaler(scaler)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = scaler.transform([test_row])[0]
            fast_result = fast_scaler(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"RobustScaler mismatch for row {i}"

    def test_maxabs_scaler(self, numeric_data):
        """Test MaxAbsScaler handler."""
        scaler = MaxAbsScaler()
        scaler.fit(numeric_data)

        fast_scaler = handle_maxabs_scaler(scaler)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = scaler.transform([test_row])[0]
            fast_result = fast_scaler(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"MaxAbsScaler mismatch for row {i}"

    @pytest.mark.parametrize("norm", ["l1", "l2", "max"])
    def test_normalizer(self, numeric_data, norm):
        """Test Normalizer handler with different norms."""
        normalizer = Normalizer(norm=norm)
        normalizer.fit(numeric_data)

        fast_normalizer = handle_normalizer(normalizer)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = normalizer.transform([test_row])[0]
            fast_result = fast_normalizer(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"Normalizer ({norm}) mismatch for row {i}"

    def test_normalizer_zero_norm(self):
        """Test Normalizer with zero norm (edge case)."""
        normalizer = Normalizer(norm="l2")
        normalizer.fit(np.array([[1, 2], [3, 4]]))

        fast_normalizer = handle_normalizer(normalizer)

        # Test with zero vector
        test_row = [0.0, 0.0]
        original_result = normalizer.transform([test_row])[0]
        fast_result = fast_normalizer(test_row)

        assert np.allclose(original_result, fast_result, rtol=1e-10)

    def test_onehot_encoder(self, categorical_data):
        """Test OneHotEncoder handler."""
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(categorical_data)

        fast_encoder = handle_onehot_encoder(encoder)

        for i in range(len(categorical_data)):
            test_row = categorical_data[i].tolist()
            original_result = encoder.transform([test_row])[0]
            fast_result = fast_encoder(test_row)

            assert np.allclose(
                original_result, fast_result
            ), f"OneHotEncoder mismatch for row {i}"

    def test_onehot_encoder_drop_first(self, categorical_data):
        """Test OneHotEncoder with drop='first'."""
        encoder = OneHotEncoder(drop="first", sparse_output=False)
        encoder.fit(categorical_data)

        fast_encoder = handle_onehot_encoder(encoder)

        for i in range(len(categorical_data)):
            test_row = categorical_data[i].tolist()
            original_result = encoder.transform([test_row])[0]
            fast_result = fast_encoder(test_row)

            assert np.allclose(
                original_result, fast_result
            ), f"OneHotEncoder (drop=first) mismatch for row {i}"

    def test_onehot_encoder_unknown_category(self, categorical_data):
        """Test OneHotEncoder with unknown category."""
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(categorical_data)

        fast_encoder = handle_onehot_encoder(encoder)

        # Test with unknown category
        test_row = ["D", "W"]  # Unknown categories
        original_result = encoder.transform([test_row])[0]
        fast_result = fast_encoder(test_row)

        assert np.allclose(original_result, fast_result)

    def test_ordinal_encoder(self, categorical_data):
        """Test OrdinalEncoder handler."""
        encoder = OrdinalEncoder()
        encoder.fit(categorical_data)

        fast_encoder = handle_ordinal_encoder(encoder)

        for i in range(len(categorical_data)):
            test_row = categorical_data[i].tolist()
            original_result = encoder.transform([test_row])[0]
            fast_result = fast_encoder(test_row)

            assert np.allclose(
                original_result, fast_result
            ), f"OrdinalEncoder mismatch for row {i}"

    def test_ordinal_encoder_unknown_category(self, categorical_data):
        """Test OrdinalEncoder with unknown category."""
        encoder = OrdinalEncoder()
        encoder.fit(categorical_data)

        fast_encoder = handle_ordinal_encoder(encoder)

        # Test with unknown category
        test_row = ["D", "W"]
        fast_result = fast_encoder(test_row)

        # Should return -1 for unknown categories
        assert fast_result == [-1.0, -1.0]

    def test_label_encoder(self, label_data):
        """Test LabelEncoder handler."""
        encoder = LabelEncoder()
        encoder.fit(label_data)

        fast_encoder = handle_label_encoder(encoder)

        for label in ["cat", "dog", "bird"]:
            original_result = encoder.transform([label])[0]
            fast_result = fast_encoder([label])

            assert fast_result == [
                float(original_result)
            ], f"LabelEncoder mismatch for label {label}"

    def test_label_encoder_single_value(self, label_data):
        """Test LabelEncoder with single value input."""
        encoder = LabelEncoder()
        encoder.fit(label_data)

        fast_encoder = handle_label_encoder(encoder)

        # Test with single value (not in list)
        original_result = encoder.transform(["cat"])[0]
        fast_result = fast_encoder("cat")

        assert fast_result == [float(original_result)]

    def test_label_encoder_unknown_label(self, label_data):
        """Test LabelEncoder with unknown label."""
        encoder = LabelEncoder()
        encoder.fit(label_data)

        fast_encoder = handle_label_encoder(encoder)

        # Test with unknown label
        fast_result = fast_encoder(["fish"])
        assert fast_result == [-1.0]

    def test_label_binarizer(self, label_data):
        """Test LabelBinarizer handler."""
        encoder = LabelBinarizer()
        encoder.fit(label_data)

        fast_encoder = handle_label_binarizer(encoder)

        for label in ["cat", "dog", "bird"]:
            original_result = encoder.transform([label])[0]
            fast_result = fast_encoder([label])

            assert np.allclose(
                original_result, fast_result
            ), f"LabelBinarizer mismatch for label {label}"

    def test_quantile_transformer_uniform(self, numeric_data):
        """Test QuantileTransformer with uniform output."""
        transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=50)
        transformer.fit(numeric_data)

        fast_transformer = handle_quantile_transformer(transformer)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = transformer.transform([test_row])[0]
            fast_result = fast_transformer(test_row)

            # Allow some tolerance due to approximation
            assert np.allclose(
                original_result, fast_result, rtol=0.1
            ), f"QuantileTransformer (uniform) mismatch for row {i}"

    def test_quantile_transformer_normal(self, numeric_data):
        """Test QuantileTransformer with normal output."""
        transformer = QuantileTransformer(output_distribution="normal", n_quantiles=50)
        transformer.fit(numeric_data)

        fast_transformer = handle_quantile_transformer(transformer)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = transformer.transform([test_row])[0]
            fast_result = fast_transformer(test_row)

            # Allow larger tolerance due to inverse normal CDF approximation
            assert np.allclose(
                original_result, fast_result, rtol=0.3
            ), f"QuantileTransformer (normal) mismatch for row {i}"

    @pytest.mark.parametrize("method", ["yeo-johnson", "box-cox"])
    def test_power_transformer(self, method):
        """Test PowerTransformer handler."""
        # Generate positive data for box-cox
        if method == "box-cox":
            X = np.random.exponential(2, (100, 3))
        else:
            X, _ = make_regression(n_samples=100, n_features=3, random_state=42)

        transformer = PowerTransformer(method=method)
        transformer.fit(X)

        fast_transformer = handle_power_transformer(transformer)

        for i in range(5):
            test_row = X[i].tolist()
            original_result = transformer.transform([test_row])[0]
            fast_result = fast_transformer(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"PowerTransformer ({method}) mismatch for row {i}"

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_binarizer(self, numeric_data, threshold):
        """Test Binarizer handler."""
        transformer = Binarizer(threshold=threshold)
        transformer.fit(numeric_data)

        fast_transformer = handle_binarizer(transformer)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            original_result = transformer.transform([test_row])[0]
            fast_result = fast_transformer(test_row)

            assert np.allclose(
                original_result, fast_result
            ), f"Binarizer (threshold={threshold}) mismatch for row {i}"

    @pytest.mark.parametrize("encode", ["onehot", "ordinal"])
    def test_kbins_discretizer(self, numeric_data, encode):
        """Test KBinsDiscretizer handler."""
        transformer = KBinsDiscretizer(n_bins=5, encode=encode, strategy="uniform")
        transformer.fit(numeric_data)

        fast_transformer = handle_kbins_discretizer(transformer)

        for i in range(5):
            test_row = numeric_data[i].tolist()
            if encode == "onehot":
                original_result = transformer.transform([test_row]).toarray()[0]
            else:
                original_result = transformer.transform([test_row])[0]
            fast_result = fast_transformer(test_row)

            assert np.allclose(
                original_result, fast_result
            ), f"KBinsDiscretizer ({encode}) mismatch for row {i}"

    # @pytest.mark.parametrize("degree", [2, 3])
    # @pytest.mark.parametrize("include_bias", [True, False])
    # def test_polynomial_features(self, degree, include_bias):
    #     """Test PolynomialFeatures handler."""
    #     # Use smaller dataset for polynomial features
    #     X = np.random.randn(20, 3)

    #     transformer = PolynomialFeatures(degree=degree, include_bias=include_bias)
    #     transformer.fit(X)

    #     fast_transformer = handle_polynomial_features(transformer)

    #     for i in range(5):
    #         test_row = X[i].tolist()
    #         original_result = transformer.transform([test_row])[0]
    #         fast_result = fast_transformer(test_row)

    #         assert np.allclose(original_result, fast_result, rtol=1e-10), \
    #             f"PolynomialFeatures (degree={degree}, bias={include_bias}) mismatch for row {i}"

    # def test_polynomial_features_degree_1(self):
    #     """Test PolynomialFeatures with degree 1 (edge case)."""
    #     X = np.random.randn(20, 3)

    #     transformer = PolynomialFeatures(degree=1)
    #     transformer.fit(X)

    #     fast_transformer = handle_polynomial_features(transformer)

    #     test_row = X[0].tolist()
    #     original_result = transformer.transform([test_row])[0]
    #     fast_result = fast_transformer(test_row)

    #     assert np.allclose(original_result, fast_result, rtol=1e-10)

    def test_target_encoder(self, categorical_data):
        """Test TargetEncoder handler."""
        # Generate target data for target encoding
        np.random.seed(42)
        y = np.random.randint(0, 2, len(categorical_data))

        encoder = TargetEncoder()
        encoder.fit(categorical_data, y)

        fast_encoder = handle_target_encoder(encoder)

        for i in range(len(categorical_data)):
            test_row = categorical_data[i].tolist()
            original_result = encoder.transform([test_row])[0]
            fast_result = fast_encoder(test_row)

            assert np.allclose(
                original_result, fast_result, rtol=1e-10
            ), f"TargetEncoder mismatch for row {i}"

    def test_target_encoder_unknown_category(self, categorical_data):
        """Test TargetEncoder with unknown category."""
        np.random.seed(42)
        y = np.random.randint(0, 2, len(categorical_data))

        encoder = TargetEncoder()
        encoder.fit(categorical_data, y)

        fast_encoder = handle_target_encoder(encoder)

        # Test with unknown category
        test_row = ["D", "W"]  # Unknown categories
        original_result = encoder.transform([test_row])[0]
        fast_result = fast_encoder(test_row)

        assert np.allclose(
            original_result, fast_result, rtol=1e-10
        ), "TargetEncoder mismatch for unknown categories"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
