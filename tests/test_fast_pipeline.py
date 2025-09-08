"""
Tests and examples for the fast pipeline compiler.
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stripje import compile_pipeline, get_supported_transformers


def create_sample_pipeline():
    """Create a sample pipeline for testing."""
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Add some categorical features
    X_df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    X_df["cat_1"] = np.random.choice(["A", "B", "C"], size=len(X_df))
    X_df["cat_2"] = np.random.choice(["X", "Y"], size=len(X_df))

    # Split features
    numeric_features = [f"num_{i}" for i in range(X.shape[1])]
    categorical_features = ["cat_1", "cat_2"]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    # Create full pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    # Fit the pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    return pipeline, X_test


def benchmark_pipelines():
    """Benchmark original vs compiled pipeline."""
    pipeline, X_test = create_sample_pipeline()

    # Compile the pipeline
    fast_predict = compile_pipeline(pipeline)

    # Convert test data to list format for fast pipeline
    X_test_lists = X_test.values.tolist()

    # Benchmark original pipeline
    start_time = time.time()
    original_predictions = []
    for row in X_test_lists:
        pred = pipeline.predict([row])[0]
        original_predictions.append(pred)
    original_time = time.time() - start_time

    # Benchmark compiled pipeline
    start_time = time.time()
    fast_predictions = []
    for row in X_test_lists:
        pred = fast_predict(row)
        fast_predictions.append(pred)
    fast_time = time.time() - start_time

    # Check accuracy
    accuracy = sum(
        1 for orig, fast in zip(original_predictions, fast_predictions) if orig == fast
    ) / len(original_predictions)

    print(
        f"Supported transformers: {[cls.__name__ for cls in get_supported_transformers()]}"
    )
    print("\nBenchmark Results:")
    print(f"Original pipeline time: {original_time:.4f}s")
    print(f"Compiled pipeline time: {fast_time:.4f}s")
    print(f"Speedup: {original_time / fast_time:.2f}x")
    print(f"Prediction accuracy: {accuracy:.4f}")

    return pipeline, fast_predict


def test_individual_transformers():
    """Test individual transformers."""
    from sklearn.tree import DecisionTreeClassifier

    # Test data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])

    print("\nTesting individual transformers:")

    # Test StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)

    from stripje.transformers.preprocessing import handle_standard_scaler

    fast_scaler = handle_standard_scaler(scaler)

    test_row = [2, 3, 4]
    original_result = scaler.transform([test_row])[0]
    fast_result = fast_scaler(test_row)

    print(f"StandardScaler - Original: {original_result}")
    print(f"StandardScaler - Fast: {fast_result}")
    print(f"StandardScaler - Match: {np.allclose(original_result, fast_result)}")

    # Test DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    from stripje.estimators.tree import handle_decision_tree_classifier

    fast_clf = handle_decision_tree_classifier(clf)

    original_pred = clf.predict([test_row])[0]
    fast_pred = fast_clf(test_row)

    print(f"DecisionTree - Original: {original_pred}")
    print(f"DecisionTree - Fast: {fast_pred}")
    print(f"DecisionTree - Match: {original_pred == fast_pred}")


def example_usage():
    """Show example usage of the fast pipeline compiler."""
    print("Creating and compiling pipeline...")
    pipeline, fast_predict = benchmark_pipelines()

    print("\nExample single-row prediction:")
    # Get a sample row
    _, X_test = create_sample_pipeline()
    sample_row = X_test.iloc[0].values.tolist()

    # Original prediction
    original_pred = pipeline.predict([sample_row])[0]

    # Fast prediction
    fast_pred = fast_predict(sample_row)

    print(f"Input row: {sample_row[:5]}...")  # Show first 5 values
    print(f"Original prediction: {original_pred}")
    print(f"Fast prediction: {fast_pred}")
    print(f"Match: {original_pred == fast_pred}")


if __name__ == "__main__":
    example_usage()
    test_individual_transformers()
