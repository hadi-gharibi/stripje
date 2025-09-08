"""
Performance test to verify that nested pipeline optimization works.
"""

import os

# Import our fast pipeline compiler
import sys
import time

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stripje.fast_pipeline import compile_pipeline


def create_deeply_nested_pipeline():
    """Create a pipeline with multiple levels of nesting."""

    # Inner pipeline
    inner_pipeline = Pipeline(
        [("scaler1", StandardScaler()), ("scaler2", MinMaxScaler())]
    )

    # Middle pipeline
    middle_pipeline = Pipeline(
        [("preprocessor", inner_pipeline), ("final_scaler", StandardScaler())]
    )

    # Outer pipeline
    outer_pipeline = Pipeline(
        [
            ("preprocessing", middle_pipeline),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    return outer_pipeline


def test_nested_performance():
    """Test performance of nested vs original pipelines."""

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    # Create and fit the nested pipeline
    pipeline = create_deeply_nested_pipeline()
    pipeline.fit(X, y)

    # Compile the pipeline
    compiled_fn = compile_pipeline(pipeline)

    # Test data
    test_rows = X[:100]

    print("=== Performance Test: Nested Pipeline ===")
    print(f"Pipeline structure: {[step[0] for step in pipeline.steps]}")

    # Test original pipeline performance
    start_time = time.time()
    original_predictions = []
    for row in test_rows:
        pred = pipeline.predict([row])[0]
        original_predictions.append(pred)
    original_time = time.time() - start_time

    # Test compiled pipeline performance
    start_time = time.time()
    compiled_predictions = []
    for row in test_rows:
        pred = compiled_fn(row.tolist())
        compiled_predictions.append(pred)
    compiled_time = time.time() - start_time

    # Check correctness
    predictions_match = all(
        o == c for o, c in zip(original_predictions, compiled_predictions)
    )

    print(f"Original pipeline time: {original_time:.4f}s")
    print(f"Compiled pipeline time: {compiled_time:.4f}s")
    print(f"Speedup: {original_time / compiled_time:.2f}x")
    print(f"Predictions match: {predictions_match}")

    return original_time, compiled_time, predictions_match


def test_recursive_compilation():
    """Test that nested pipelines are compiled recursively."""

    # Create sample data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # Create nested pipeline
    inner = Pipeline([("scaler", StandardScaler())])
    outer = Pipeline(
        [("preprocessor", inner), ("classifier", LogisticRegression(random_state=42))]
    )

    outer.fit(X, y)

    print("\n=== Recursive Compilation Test ===")

    # Compile and test
    compiled_fn = compile_pipeline(outer)

    # Test a few predictions
    for i in range(3):
        test_row = X[i]
        original_pred = outer.predict([test_row])[0]
        compiled_pred = compiled_fn(test_row.tolist())
        print(
            f"Row {i}: Original={original_pred}, Compiled={compiled_pred}, Match={original_pred == compiled_pred}"
        )


if __name__ == "__main__":
    try:
        test_nested_performance()
        test_recursive_compilation()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
