"""
Integration tests for the entire stripje library.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from stripje import compile_pipeline, get_supported_transformers


class TestIntegration:
    """Integration tests for the complete library."""

    def test_end_to_end_classification_workflow(self):
        """Test complete classification workflow from data to prediction."""
        # Load real dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Create complex pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selection", SelectKBest(score_func=f_classif, k=3)),
                ("pca", PCA(n_components=2)),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Compile pipeline
        fast_predict = compile_pipeline(pipeline)

        # Test predictions on all test samples
        correct_predictions = 0
        total_predictions = len(X_test)

        for i in range(total_predictions):
            test_row = X_test[i].tolist()

            # Get predictions from both pipelines
            original_pred = pipeline.predict([test_row])[0]
            fast_pred = fast_predict(test_row)

            # Verify they match
            assert (
                original_pred == fast_pred
            ), f"Prediction mismatch at sample {i}: {original_pred} vs {fast_pred}"

            # Check against true label
            if original_pred == y_test[i]:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"\nEnd-to-end classification accuracy: {accuracy:.3f}")

        # Should achieve reasonable accuracy on iris dataset
        assert accuracy > 0.8, f"Accuracy too low: {accuracy}"

    def test_end_to_end_regression_workflow(self):
        """Test complete regression workflow."""
        # Generate regression data
        X, y = make_regression(
            n_samples=200, n_features=10, n_informative=7, noise=0.1, random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create regression pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selection", SelectKBest(k=7)),
                ("regressor", LinearRegression()),
            ]
        )

        pipeline.fit(X_train, y_train)
        fast_predict = compile_pipeline(pipeline)

        # Test predictions and calculate MSE
        mse_original = 0
        mse_fast = 0

        for i in range(len(X_test)):
            test_row = X_test[i].tolist()
            true_value = y_test[i]

            original_pred = pipeline.predict([test_row])[0]
            fast_pred = fast_predict(test_row)

            # Verify predictions match
            assert np.isclose(
                original_pred, fast_pred, rtol=1e-10
            ), f"Regression prediction mismatch at sample {i}"

            mse_original += (original_pred - true_value) ** 2
            mse_fast += (fast_pred - true_value) ** 2

        mse_original /= len(X_test)
        mse_fast /= len(X_test)

        print(f"\nRegression MSE (original): {mse_original:.6f}")
        print(f"Regression MSE (fast): {mse_fast:.6f}")

        # MSE should be identical (within numerical precision)
        assert np.isclose(mse_original, mse_fast, rtol=1e-10)

    def test_mixed_data_types_workflow(self):
        """Test workflow with mixed numeric and categorical data."""
        # Create mixed dataset
        X_num, y = make_classification(
            n_samples=300, n_features=8, n_informative=6, n_redundant=1, random_state=42
        )

        # Create DataFrame with mixed types
        df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])
        df["cat_size"] = np.random.choice(["Small", "Medium", "Large"], size=len(df))
        df["cat_color"] = np.random.choice(
            ["Red", "Blue", "Green", "Yellow"], size=len(df)
        )
        df["cat_type"] = np.random.choice(["A", "B"], size=len(df))

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.3, random_state=42, stratify=y
        )

        # Define feature groups
        numeric_features = [col for col in df.columns if col.startswith("num_")]
        categorical_features = [col for col in df.columns if col.startswith("cat_")]

        # Create comprehensive preprocessing pipeline
        preprocessor = ColumnTransformer(
            [
                ("num_scale", StandardScaler(), numeric_features[:4]),
                ("num_minmax", MinMaxScaler(), numeric_features[4:]),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X_train, y_train)
        fast_predict = compile_pipeline(pipeline)

        # Test with different input formats
        for i in range(min(20, len(X_test))):
            # Test with dictionary input
            test_row_dict = X_test.iloc[i].to_dict()

            # Pass as DataFrame to maintain column names
            original_pred = pipeline.predict(X_test.iloc[[i]])[0]
            fast_pred = fast_predict(test_row_dict)

            assert (
                original_pred == fast_pred
            ), f"Mixed data prediction mismatch at sample {i}"

        print("Mixed data types workflow completed successfully")

    def test_all_supported_transformers_integration(self):
        """Test that all supported transformers work in a pipeline."""
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import (
            Normalizer,
            RobustScaler,
        )

        # Get list of supported transformers
        supported = get_supported_transformers()
        transformer_names = [cls.__name__ for cls in supported]

        print(f"\nSupported transformers: {len(transformer_names)}")
        print(f"Types: {', '.join(sorted(transformer_names))}")

        # Test a subset of transformers in sequence
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=8, random_state=42
        )

        # Ensure positive values for certain transformers
        X = np.abs(X) + 0.1

        # Create a pipeline with multiple different transformers
        pipeline = Pipeline(
            [
                ("robust_scaler", RobustScaler()),
                ("variance_threshold", VarianceThreshold(threshold=0.01)),
                ("normalizer", Normalizer()),
                ("feature_selection", SelectKBest(k=5)),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        # Test predictions
        test_row = X[0].tolist()
        original_pred = pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred
        print("All transformers integration test passed")

    def test_real_world_scenario_customer_data(self):
        """Test with a real-world-like customer dataset scenario."""
        np.random.seed(42)

        # Simulate customer data
        n_customers = 500

        # Create realistic customer features
        age = np.random.normal(40, 15, n_customers).clip(18, 80)
        income = np.random.lognormal(10, 0.5, n_customers).clip(20000, 200000)
        spending_score = np.random.uniform(0, 100, n_customers)

        # Categorical features
        education = np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"],
            n_customers,
            p=[0.3, 0.4, 0.2, 0.1],
        )
        city_size = np.random.choice(
            ["Small", "Medium", "Large"], n_customers, p=[0.3, 0.4, 0.3]
        )

        # Create target (will customer buy premium product?)
        # More likely if high income, high spending score, higher education
        buy_probability = (
            (income / 100000) * 0.3
            + (spending_score / 100) * 0.4
            + (age / 80) * 0.1
            + np.where(
                education == "PhD",
                0.2,
                np.where(
                    education == "Master",
                    0.15,
                    np.where(education == "Bachelor", 0.1, 0.05),
                ),
            )
        )

        will_buy = np.random.binomial(1, buy_probability.clip(0, 1), n_customers)

        # Create DataFrame
        customer_df = pd.DataFrame(
            {
                "age": age,
                "income": income,
                "spending_score": spending_score,
                "education": education,
                "city_size": city_size,
                "will_buy": will_buy,
            }
        )

        # Split features and target
        X = customer_df.drop("will_buy", axis=1)
        y = customer_df["will_buy"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Create realistic ML pipeline
        numeric_features = ["age", "income", "spending_score"]
        categorical_features = ["education", "city_size"]

        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first"), categorical_features),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("feature_selection", SelectKBest(score_func=f_classif, k=8)),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X_train, y_train)
        fast_predict = compile_pipeline(pipeline)

        # Test predictions on new customers
        correct_predictions = 0

        for i in range(len(X_test)):
            customer_data = X_test.iloc[i].to_dict()
            true_label = y_test.iloc[i]

            # Pass as DataFrame to maintain column names
            original_pred = pipeline.predict(X_test.iloc[[i]])[0]
            fast_pred = fast_predict(customer_data)

            # Verify predictions match
            assert (
                original_pred == fast_pred
            ), f"Customer prediction mismatch for customer {i}"

            if original_pred == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(X_test)
        print(f"\nCustomer prediction accuracy: {accuracy:.3f}")

        # Should achieve reasonable accuracy (relaxed threshold for test stability)
        assert accuracy > 0.55, f"Customer model accuracy too low: {accuracy}"

        print("Real-world customer scenario completed successfully")

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)

        # Test with minimal pipeline
        minimal_pipeline = Pipeline(
            [("classifier", LogisticRegression(random_state=42, max_iter=1000))]
        )
        minimal_pipeline.fit(X, y)

        fast_predict = compile_pipeline(minimal_pipeline)

        test_row = X[0].tolist()
        original_pred = minimal_pipeline.predict([test_row])[0]
        fast_pred = fast_predict(test_row)

        assert original_pred == fast_pred

        # Test with single feature
        X_single = X[:, :1]
        single_feature_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        single_feature_pipeline.fit(X_single, y)

        fast_predict_single = compile_pipeline(single_feature_pipeline)

        test_row_single = [X_single[0, 0]]
        original_pred_single = single_feature_pipeline.predict([test_row_single])[0]
        fast_pred_single = fast_predict_single(test_row_single)

        assert original_pred_single == fast_pred_single

        print("Error handling and edge cases test passed")

    def test_library_consistency_check(self):
        """Final consistency check across the entire library."""
        # Test that all major components work together
        from stripje.registry import STEP_HANDLERS

        print("\nLibrary consistency check:")
        print(f"Total registered handlers: {len(STEP_HANDLERS)}")
        print(f"Handler types: {list(STEP_HANDLERS.keys())[:5]}...")  # Show first 5

        # Verify that we can create and compile a complex pipeline
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selection", SelectKBest(k=5)),
                ("pca", PCA(n_components=3)),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        # Test multiple predictions
        predictions_match = True
        for i in range(10):
            test_row = X[i].tolist()
            original_pred = pipeline.predict([test_row])[0]
            fast_pred = fast_predict(test_row)

            if original_pred != fast_pred:
                predictions_match = False
                break

        assert predictions_match, "Library consistency check failed"
        print("Library consistency check passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output
