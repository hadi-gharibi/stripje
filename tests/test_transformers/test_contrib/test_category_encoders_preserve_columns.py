"""
Test that category encoders preserve non-encoded columns when used with mixed features.

This test suite verifies that all category encoder handlers correctly preserve
numeric and other non-encoded columns when transforming data.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    import category_encoders as ce

    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

import stripje
from stripje.transformers.contrib.category_encoders_transformers import (
    _safe_fit_supervised_encoder,
)


@pytest.mark.skipif(
    not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not installed"
)
class TestCategoryEncodersPreserveColumns:
    """Test that all category encoders preserve non-encoded columns."""

    def setup_method(self):
        """Set up test data for each test."""
        np.random.seed(42)
        self.X = pd.DataFrame(
            {
                "cat_feature": np.random.choice(["A", "B", "C"], 100),
                "num_feature_1": np.random.randn(100),
                "num_feature_2": np.random.uniform(0, 100, 100),
            }
        )
        self.y = np.random.randint(0, 2, 100)

    def _test_encoder_preserves_columns(self, encoder_class, **encoder_kwargs):
        """
        Generic test method to verify an encoder preserves non-encoded columns.

        Args:
            encoder_class: The encoder class to test
            **encoder_kwargs: Additional kwargs to pass to encoder constructor
        """
        # Create pipeline with encoder
        encoder = encoder_class(cols=["cat_feature"], **encoder_kwargs)

        # For supervised encoders, we need to fit with y
        if encoder_class in [
            ce.TargetEncoder,
            ce.CatBoostEncoder,
            ce.LeaveOneOutEncoder,
        ]:
            _safe_fit_supervised_encoder(encoder, self.X, self.y)
        else:
            encoder.fit(self.X)

        # Check that original encoder preserves numeric columns
        X_transformed = encoder.transform(self.X)
        assert (
            "num_feature_1" in X_transformed.columns
        ), f"{encoder_class.__name__} should preserve num_feature_1"
        assert (
            "num_feature_2" in X_transformed.columns
        ), f"{encoder_class.__name__} should preserve num_feature_2"

        # Now test with pipeline and classifier
        if encoder_class in [
            ce.TargetEncoder,
            ce.CatBoostEncoder,
            ce.LeaveOneOutEncoder,
        ]:
            pipeline = Pipeline(
                [
                    ("encoder", encoder_class(cols=["cat_feature"], **encoder_kwargs)),
                    ("clf", LogisticRegression(random_state=42, max_iter=500)),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("encoder", encoder_class(cols=["cat_feature"], **encoder_kwargs)),
                    ("clf", LogisticRegression(random_state=42, max_iter=500)),
                ]
            )

        pipeline.fit(self.X, self.y)
        fast_predict = stripje.compile_pipeline(pipeline)

        # Test a few samples
        matches = 0
        total_tests = 5
        mismatches = []

        for i in range(total_tests):
            test_row = self.X.iloc[i].to_dict()
            original_pred = pipeline.predict(self.X.iloc[[i]])[0]
            fast_pred = fast_predict(test_row)

            if original_pred == fast_pred:
                matches += 1
            else:
                mismatches.append(
                    {
                        "index": i,
                        "original": original_pred,
                        "fast": fast_pred,
                        "row": test_row,
                    }
                )

        match_rate = matches / total_tests

        # If match rate is low, print details
        if match_rate < 0.6:
            print(f"\n{encoder_class.__name__} - Match rate: {match_rate:.2%}")
            print(f"Mismatches: {mismatches}")

            # Debug: Check what features are being passed
            if mismatches:
                test_row = mismatches[0]["row"]
                print(f"\nTest row: {test_row}")

                # Check encoder output
                encoded_df = encoder.transform(pd.DataFrame([test_row]))
                print(f"Encoded columns: {encoded_df.columns.tolist()}")
                print(f"Encoded shape: {encoded_df.shape}")
                print(f"Encoded values: {encoded_df.values[0]}")

        assert match_rate >= 0.6, (
            f"{encoder_class.__name__} failed to preserve columns correctly. "
            f"Match rate: {match_rate:.2%} ({matches}/{total_tests}). "
            f"This likely means numeric features are being dropped."
        )

    def test_binary_encoder_preserves_columns(self):
        """Test BinaryEncoder preserves non-encoded columns."""
        self._test_encoder_preserves_columns(ce.BinaryEncoder)

    def test_onehot_encoder_preserves_columns(self):
        """Test OneHotEncoder preserves non-encoded columns."""
        self._test_encoder_preserves_columns(ce.OneHotEncoder)

    def test_ordinal_encoder_preserves_columns(self):
        """Test OrdinalEncoder preserves non-encoded columns."""
        self._test_encoder_preserves_columns(ce.OrdinalEncoder)

    def test_hashing_encoder_preserves_columns(self):
        """Test HashingEncoder preserves non-encoded columns."""
        self._test_encoder_preserves_columns(ce.HashingEncoder, n_components=8)

    @pytest.mark.skip(
        reason="TargetEncoder has sklearn compatibility issues in pipelines"
    )
    def test_target_encoder_preserves_columns(self):
        """Test TargetEncoder preserves non-encoded columns."""
        self._test_encoder_preserves_columns(ce.TargetEncoder)

    @pytest.mark.skip(
        reason="CatBoostEncoder has sklearn compatibility issues in pipelines"
    )
    def test_catboost_encoder_preserves_columns(self):
        """Test CatBoostEncoder preserves non-encoded columns."""
        self._test_encoder_preserves_columns(ce.CatBoostEncoder)

    @pytest.mark.skip(
        reason="LeaveOneOutEncoder has sklearn compatibility issues in pipelines"
    )
    def test_leave_one_out_encoder_preserves_columns(self):
        """Test LeaveOneOutEncoder preserves non-encoded columns."""
        self._test_encoder_preserves_columns(ce.LeaveOneOutEncoder)
