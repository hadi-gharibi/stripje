"""
Test script to verify behavior with nested pipelines.
"""

import os

# Import our fast pipeline compiler
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stripje.fast_pipeline import compile_pipeline
from stripje.registry import get_handler


def test_nested_pipeline_behavior():
    """Test what happens with nested pipelines."""

    # Create some sample data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # Create a nested pipeline
    inner_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    outer_pipeline = Pipeline(
        [
            ("preprocessor", StandardScaler()),  # This should have a handler
            ("model", inner_pipeline),  # This nested pipeline won't have a handler
        ]
    )

    # Fit the pipeline
    outer_pipeline.fit(X, y)

    print("=== Testing Nested Pipeline Behavior ===")

    # Check if Pipeline type has a handler
    pipeline_handler = get_handler(Pipeline)
    print(f"Handler for Pipeline type: {pipeline_handler}")

    # Try to compile the outer pipeline
    print("\nCompiling outer pipeline...")
    compiled_fn = compile_pipeline(outer_pipeline)

    # Test single row prediction
    test_row = X[0]
    print(f"Test row: {test_row}")

    # Compare predictions
    original_pred = outer_pipeline.predict([test_row])[0]
    compiled_pred = compiled_fn(test_row.tolist())

    print(f"Original prediction: {original_pred}")
    print(f"Compiled prediction: {compiled_pred}")
    print(f"Predictions match: {original_pred == compiled_pred}")

    return original_pred, compiled_pred


def test_simple_pipeline_for_comparison():
    """Test a simple pipeline without nesting for comparison."""

    # Create some sample data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # Create a simple pipeline (no nesting)
    simple_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    # Fit the pipeline
    simple_pipeline.fit(X, y)

    print("\n=== Testing Simple Pipeline for Comparison ===")

    # Compile the pipeline
    print("Compiling simple pipeline...")
    compiled_fn = compile_pipeline(simple_pipeline)

    # Test single row prediction
    test_row = X[0]
    print(f"Test row: {test_row}")

    # Compare predictions
    original_pred = simple_pipeline.predict([test_row])[0]
    compiled_pred = compiled_fn(test_row.tolist())

    print(f"Original prediction: {original_pred}")
    print(f"Compiled prediction: {compiled_pred}")
    print(f"Predictions match: {original_pred == compiled_pred}")

    return original_pred, compiled_pred


if __name__ == "__main__":
    try:
        test_nested_pipeline_behavior()
        test_simple_pipeline_for_comparison()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
