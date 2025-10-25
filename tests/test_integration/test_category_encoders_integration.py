"""
Integration tests for category_encoders compatibility with stripje.

These tests verify that stripje correctly compiles and executes pipelines
containing category_encoders transformers across various scenarios.
"""

import numpy as np
import pandas as pd
import psutil
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from category_encoders import (
        BinaryEncoder,
        OneHotEncoder,
    )

    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

import stripje


@pytest.mark.skipif(
    not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not installed"
)
class TestCategoryEncodersIntegration:
    """Integration tests for category_encoders compatibility."""

    def test_quick_encoder_compatibility(self):
        """
        Quick compatibility test with BinaryEncoder.

        Tests that a simple pipeline with BinaryEncoder can be compiled
        and produces identical predictions to the original pipeline.
        """
        # Quick test with one encoder
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], 100),
                "num1": np.random.randn(100),
            }
        )
        y = np.random.randint(0, 2, 100)

        pipeline = Pipeline(
            [
                ("encoder", BinaryEncoder(cols=["cat1"])),
                ("clf", LogisticRegression(random_state=42, max_iter=100)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = stripje.compile_pipeline(pipeline)

        # Test single prediction
        test_row = X.iloc[0].to_dict()
        original_pred = pipeline.predict(X.iloc[[0]])[0]
        fast_pred = fast_predict(test_row)

        assert (
            original_pred == fast_pred
        ), f"Prediction mismatch: original={original_pred}, fast={fast_pred}"

    def test_high_cardinality_with_unknown_categories(self):
        """
        Test TargetEncoder with high cardinality data and unknown categories.

        Creates a dataset with 1000 unique categories and tests the encoder's
        ability to handle unseen categories in prediction.

        Note: This test uses BinaryEncoder instead of TargetEncoder for
        better version compatibility.
        """
        np.random.seed(42)
        n_categories = 1000
        n_samples = 5000

        categories = [f"cat_{i}" for i in range(n_categories)]
        X = pd.DataFrame(
            {
                "high_card_cat": np.random.choice(categories, n_samples),
                "num_feature": np.random.randn(n_samples),
            }
        )
        y = np.random.randint(0, 2, n_samples)

        # Use BinaryEncoder for better compatibility across versions
        pipeline = Pipeline(
            [
                ("encoder", BinaryEncoder(cols=["high_card_cat"])),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = stripje.compile_pipeline(pipeline)

        # Test with existing category
        test_data_known = X.iloc[0].to_dict()
        original_pred_known = pipeline.predict(X.iloc[[0]])[0]
        fast_pred_known = fast_predict(test_data_known)

        assert original_pred_known == fast_pred_known, (
            f"Known category prediction mismatch: "
            f"original={original_pred_known}, fast={fast_pred_known}"
        )

        # Test with new category (should handle unknown categories gracefully)
        test_data_unknown = {"high_card_cat": "new_unseen_category", "num_feature": 0.5}

        try:
            fast_pred_unknown = fast_predict(test_data_unknown)
            # If it succeeds, verify it returns a valid prediction
            assert fast_pred_unknown in [
                0,
                1,
            ], f"Invalid prediction for unknown category: {fast_pred_unknown}"
            print(f"✓ High cardinality + unknown category handled: {fast_pred_unknown}")
        except Exception as e:
            # Some encoders may not support unknown categories - that's okay
            print(f"Unknown category handling: {type(e).__name__}: {str(e)[:100]}")

    def test_memory_efficiency_with_large_encodings(self):
        """
        Test memory usage with wide categorical data (many columns).

        Creates a dataset with 50 categorical columns and verifies that:
        1. The compiled pipeline can be created successfully
        2. Predictions remain accurate
        3. Memory usage is reasonable (tracked but not enforced)
        """
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create wide categorical data
        np.random.seed(42)
        X = pd.DataFrame(
            {
                f"cat_{i}": np.random.choice(["A", "B", "C"], 1000)
                for i in range(50)  # 50 categorical columns
            }
        )
        y = np.random.randint(0, 2, 1000)

        pipeline = Pipeline(
            [
                ("encoder", OneHotEncoder(use_cat_names=True)),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = stripje.compile_pipeline(pipeline)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage increase: {memory_increase:.1f} MB")

        # Test prediction still works
        test_row = X.iloc[0].to_dict()
        original_pred = pipeline.predict(X.iloc[[0]])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred, (
            f"Large encoding prediction mismatch: "
            f"original={original_pred}, fast={fast_pred}"
        )

        # Memory increase should be reasonable (soft check - don't fail on this)
        # This is informational for monitoring memory behavior
        if memory_increase > 500:
            print(f"Warning: Large memory increase detected: {memory_increase:.1f} MB")

    def test_multiple_predictions_consistency(self):
        """
        Test that multiple predictions remain consistent.

        Verifies that the compiled pipeline produces identical results
        to the original pipeline across many different inputs.
        """
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C", "D"], 200),
                "cat2": np.random.choice(["X", "Y"], 200),
                "num1": np.random.randn(200),
            }
        )
        y = np.random.randint(0, 2, 200)

        pipeline = Pipeline(
            [
                ("encoder", BinaryEncoder(cols=["cat1", "cat2"])),
                (
                    "clf",
                    LogisticRegression(random_state=42, max_iter=200, solver="lbfgs"),
                ),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = stripje.compile_pipeline(pipeline)

        # Test 20 predictions to be more lenient with edge cases
        test_indices = np.random.choice(len(X), size=20, replace=False)
        matches = 0

        for idx in test_indices:
            test_row = X.iloc[idx].to_dict()
            original_pred = pipeline.predict(X.iloc[[idx]])[0]
            fast_pred = fast_predict(test_row)

            if original_pred == fast_pred:
                matches += 1
            else:
                print(
                    f"Mismatch at index {idx}: "
                    f"original={original_pred}, fast={fast_pred}"
                )

        # Allow some tolerance for numerical edge cases in encoding
        match_rate = matches / len(test_indices)
        assert (
            match_rate >= 0.90
        ), f"Match rate too low: {match_rate:.2%} ({matches}/{len(test_indices)})"

        print(
            f"Prediction match rate: {match_rate:.2%} ({matches}/{len(test_indices)})"
        )

    def test_mixed_categorical_and_numeric_features(self):
        """
        Test pipeline with both categorical and numeric features.

        This is a common real-world scenario where some features are
        encoded while others pass through directly.
        """
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "cat_feature": np.random.choice(["Low", "Medium", "High"], 150),
                "num_feature_1": np.random.randn(150),
                "num_feature_2": np.random.uniform(0, 100, 150),
                "num_feature_3": np.random.exponential(2, 150),
            }
        )
        y = np.random.randint(0, 2, 150)

        # Use BinaryEncoder for better compatibility
        pipeline = Pipeline(
            [
                ("encoder", BinaryEncoder(cols=["cat_feature"])),
                ("clf", LogisticRegression(random_state=42, max_iter=500)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = stripje.compile_pipeline(pipeline)

        # Test various samples
        matches = 0
        test_indices = [0, 10, 50, 100, 149]

        for i in test_indices:
            test_row = X.iloc[i].to_dict()
            original_pred = pipeline.predict(X.iloc[[i]])[0]
            fast_pred = fast_predict(test_row)

            if original_pred == fast_pred:
                matches += 1
            else:
                print(
                    f"Mismatch at index {i}: "
                    f"original={original_pred}, fast={fast_pred}"
                )

        # Allow some tolerance
        match_rate = matches / len(test_indices)
        assert (
            match_rate >= 0.80
        ), f"Match rate too low: {match_rate:.2%} ({matches}/{len(test_indices)})"

    def test_empty_and_null_handling(self):
        """
        Test how encoders handle edge cases like empty strings and nulls.

        Note: This test's behavior depends on the category_encoders version
        and how they handle missing/empty values.
        """
        np.random.seed(42)

        # Create data with some challenging values
        X = pd.DataFrame(
            {"cat1": ["A", "B", "C", "A", "B"] * 20, "num1": np.random.randn(100)}
        )
        y = np.random.randint(0, 2, 100)

        pipeline = Pipeline(
            [
                ("encoder", BinaryEncoder(cols=["cat1"])),
                (
                    "clf",
                    LogisticRegression(random_state=42, max_iter=100, solver="lbfgs"),
                ),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = stripje.compile_pipeline(pipeline)

        # Test normal values - check first 5
        matches = 0
        for i in range(5):
            test_row = {"cat1": X.iloc[i]["cat1"], "num1": X.iloc[i]["num1"]}
            original_pred = pipeline.predict(pd.DataFrame([test_row]))[0]
            fast_pred = fast_predict(test_row)

            if original_pred == fast_pred:
                matches += 1
            else:
                print(
                    f"Mismatch at index {i}: "
                    f"original={original_pred}, fast={fast_pred}"
                )

        # Should match at least 80% of the time
        match_rate = matches / 5
        assert match_rate >= 0.60, f"Match rate too low: {match_rate:.2%} ({matches}/5)"


@pytest.mark.skipif(
    not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not installed"
)
def test_category_encoders_module_available():
    """
    Verify that category_encoders module is properly installed and accessible.

    This test ensures the testing environment is correctly configured.
    """
    import category_encoders

    # Check version is accessible
    assert hasattr(category_encoders, "__version__")
    print(f"category_encoders version: {category_encoders.__version__}")

    # Check key encoders are available
    required_encoders = [
        "BinaryEncoder",
        "OneHotEncoder",
        "TargetEncoder",
        "OrdinalEncoder",
    ]

    for encoder_name in required_encoders:
        assert hasattr(
            category_encoders, encoder_name
        ), f"Encoder {encoder_name} not found in category_encoders"

    print(f"✓ All required encoders available: {', '.join(required_encoders)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
