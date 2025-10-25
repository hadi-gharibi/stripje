"""
Additional tests to improve fast_pipeline.py coverage.
"""

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stripje import compile_pipeline


class TestFastPipelineCoverage:
    """Tests to cover missing branches in fast_pipeline.py."""

    def test_column_transformer_with_passthrough(self):
        """Test ColumnTransformer with passthrough columns."""
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        y = np.array([0, 1, 0])

        # Create a ColumnTransformer with passthrough
        ct = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), [0, 1]),
                ("passthrough", "passthrough", [2, 3]),
            ]
        )

        pipeline = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)

        # Test prediction
        test_row = [2, 3, 4, 5]
        original_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred

    def test_column_transformer_with_drop(self):
        """Test ColumnTransformer with dropped columns."""
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        y = np.array([0, 1, 0])

        # Create a ColumnTransformer with drop
        ct = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), [0, 1]),
                ("drop", "drop", [2, 3]),
            ]
        )

        pipeline = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)

        # Test prediction
        test_row = [2, 3, 4, 5]
        original_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred

    def test_column_transformer_dict_input(self):
        """Test ColumnTransformer with dictionary input."""
        import pandas as pd

        X = pd.DataFrame({"a": [1, 5, 9], "b": [2, 6, 10], "c": [3, 7, 11]})
        y = np.array([0, 1, 0])

        # Create a ColumnTransformer
        ct = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), ["a", "b"]),
                ("passthrough", "passthrough", ["c"]),
            ]
        )

        pipeline = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)

        # Test with dictionary input
        test_dict = {"a": 2, "b": 3, "c": 4}
        test_df = pd.DataFrame([test_dict])

        original_pred = pipeline.predict(test_df)[0]
        fast_pred = fast_predict(test_dict)

        assert original_pred == fast_pred

    def test_nested_pipeline(self):
        """Test nested Pipeline handling."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 0])

        # Create nested pipeline
        inner_pipeline = Pipeline([("scaler", StandardScaler())])

        outer_pipeline = Pipeline(
            [
                ("inner", inner_pipeline),
                ("clf", LogisticRegression()),
            ]
        )

        outer_pipeline.fit(X, y)

        fast_predict = compile_pipeline(outer_pipeline)

        # Test prediction
        test_row = [2, 3, 4]
        original_pred = outer_pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred

    def test_ridge_classifier_with_2d_coef(self):
        """Test RidgeClassifier binary case with 2D coefficients."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]])
        y = np.array([0, 1, 0, 1])

        # RidgeClassifier sometimes has 2D coef even for binary
        clf = RidgeClassifier()
        clf.fit(X, y)

        from stripje.estimators.linear import handle_ridge_classifier

        fast_clf = handle_ridge_classifier(clf)

        # Test prediction
        test_row = [2, 3, 4]
        original_pred = clf.predict([test_row])[0]
        fast_pred = fast_clf(test_row)

        assert original_pred == fast_pred

    def test_column_transformer_single_string_column(self):
        """Test ColumnTransformer with single string column name."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 0])

        # Create a ColumnTransformer with single column as string
        ct = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), [0]),
                ("passthrough", "passthrough", [1, 2]),
            ],
            remainder="drop",
        )

        pipeline = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
        pipeline.fit(X, y)

        fast_predict = compile_pipeline(pipeline)

        # Test prediction with list input
        test_row = [2, 3, 4]
        original_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
