"""
Performance benchmarks and speed tests for the fast pipeline compiler.
"""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stripje.fast_pipeline import compile_pipeline


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks."""

    @pytest.fixture
    def large_dataset(self):
        """Generate a larger dataset for performance testing."""
        X, y = make_classification(
            n_samples=1000,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            random_state=42,
        )

        # Convert to DataFrame and add categorical features
        df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
        df["cat_1"] = np.random.choice(["A", "B", "C", "D"], size=len(df))
        df["cat_2"] = np.random.choice(["X", "Y", "Z"], size=len(df))

        return df, y

    @pytest.fixture
    def complex_pipeline(self, large_dataset):
        """Create a complex pipeline for testing."""
        df, y = large_dataset

        numeric_features = [col for col in df.columns if col.startswith("num_")]
        categorical_features = [col for col in df.columns if col.startswith("cat_")]

        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first"), categorical_features),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(df, y)
        return pipeline, df

    def test_single_prediction_speed(self, complex_pipeline):
        """Test speed of single predictions."""
        pipeline, df = complex_pipeline

        # Compile the pipeline
        fast_predict = compile_pipeline(pipeline)

        # Prepare test data
        test_rows = [df.iloc[i].to_dict() for i in range(100)]

        # Benchmark original pipeline
        start_time = time.time()
        original_predictions = []
        for row_dict in test_rows:
            row_df = pd.DataFrame([row_dict])
            pred = pipeline.predict(row_df)[0]
            original_predictions.append(pred)
        original_time = time.time() - start_time

        # Benchmark compiled pipeline
        start_time = time.time()
        fast_predictions = []
        for row_dict in test_rows:
            pred = fast_predict(row_dict)
            fast_predictions.append(pred)
        fast_time = time.time() - start_time

        # Verify accuracy
        accuracy = sum(
            1
            for orig, fast in zip(original_predictions, fast_predictions)
            if orig == fast
        ) / len(original_predictions)

        speedup = original_time / fast_time if fast_time > 0 else float("inf")

        print("\nSingle Prediction Benchmark:")
        print(f"Original time: {original_time:.4f}s")
        print(f"Fast time: {fast_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Accuracy: {accuracy:.4f}")

        # Assertions
        assert accuracy >= 0.99, "Fast pipeline should maintain high accuracy"
        assert speedup > 1.0, "Fast pipeline should be faster than original"
        assert fast_time < original_time, "Fast pipeline should take less time"

    def test_batch_prediction_speed(self, complex_pipeline):
        """Test speed comparison for batch predictions."""
        pipeline, df = complex_pipeline
        fast_predict = compile_pipeline(pipeline)

        # Test with larger batch
        n_predictions = 1000
        test_data = df.sample(n=n_predictions, random_state=42)
        test_rows = [test_data.iloc[i].to_dict() for i in range(n_predictions)]

        # Benchmark batch predictions
        start_time = time.time()
        for row_dict in test_rows:
            row_df = pd.DataFrame([row_dict])
            pipeline.predict(row_df)[0]
        original_batch_time = time.time() - start_time

        start_time = time.time()
        for row_dict in test_rows:
            fast_predict(row_dict)
        fast_batch_time = time.time() - start_time

        speedup = (
            original_batch_time / fast_batch_time
            if fast_batch_time > 0
            else float("inf")
        )

        print(f"\nBatch Prediction Benchmark ({n_predictions} predictions):")
        print(f"Original time: {original_batch_time:.4f}s")
        print(f"Fast time: {fast_batch_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(
            f"Predictions per second (original): {n_predictions / original_batch_time:.0f}"
        )
        print(f"Predictions per second (fast): {n_predictions / fast_batch_time:.0f}")

        assert speedup > 1.0

    @pytest.mark.parametrize("n_features", [5, 10, 20])
    def test_speed_vs_feature_count(self, n_features):
        """Test how speed scales with number of features."""
        X, y = make_classification(
            n_samples=500,
            n_features=n_features,
            n_informative=min(n_features - 1, 10),
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        pipeline.fit(X, y)
        fast_predict = compile_pipeline(pipeline)

        # Test speed with 100 predictions
        test_rows = X[:100].tolist()

        start_time = time.time()
        for row in test_rows:
            pipeline.predict([row])[0]
        original_time = time.time() - start_time

        start_time = time.time()
        for row in test_rows:
            fast_predict(row)
        fast_time = time.time() - start_time

        speedup = original_time / fast_time if fast_time > 0 else float("inf")

        print(f"\nFeature Count Benchmark ({n_features} features):")
        print(f"Speedup: {speedup:.2f}x")

        assert speedup > 1.0

    def test_memory_usage_comparison(self, complex_pipeline):
        """Test memory usage of compiled vs original pipeline."""
        import sys

        pipeline, df = complex_pipeline

        # Measure memory before compilation
        original_size = sys.getsizeof(pipeline)

        # Compile pipeline
        fast_predict = compile_pipeline(pipeline)
        fast_predict_size = sys.getsizeof(fast_predict)

        print("\nMemory Usage Comparison:")
        print(f"Original pipeline: {original_size} bytes")
        print(f"Fast predict function: {fast_predict_size} bytes")

        # The compiled function should generally be smaller than the full pipeline
        # Note: This test is informational, actual memory usage may vary

    def test_compilation_time(self, large_dataset):
        """Test how long it takes to compile different pipeline types."""
        df, y = large_dataset

        pipelines = {
            "simple": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=1000)),
                ]
            ),
            "with_feature_selection": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=1000)),
                ]
            ),
        }

        for name, pipeline in pipelines.items():
            # Fit pipeline
            if "cat" in name:
                pipeline.fit(df, y)
            else:
                numeric_data = df.select_dtypes(include=[np.number])
                pipeline.fit(numeric_data, y)

            # Measure compilation time
            start_time = time.time()
            compile_pipeline(pipeline)
            compilation_time = time.time() - start_time

            print(f"\nCompilation time for {name}: {compilation_time:.4f}s")

            # Compilation should be fast (under 1 second for most cases)
            assert compilation_time < 1.0, f"Compilation too slow for {name}"

    def test_concurrent_predictions(self, complex_pipeline):
        """Test thread safety and concurrent predictions."""
        import queue
        import threading

        pipeline, df = complex_pipeline
        fast_predict = compile_pipeline(pipeline)

        # Prepare test data
        test_rows = [df.iloc[i].to_dict() for i in range(50)]
        results_queue = queue.Queue()

        def make_predictions(thread_id):
            """Function to run predictions in a thread."""
            thread_results = []
            for i, row in enumerate(test_rows):
                pred = fast_predict(row)
                thread_results.append((thread_id, i, pred))
            results_queue.put(thread_results)

        # Run predictions in multiple threads
        threads = []
        n_threads = 4

        start_time = time.time()
        for thread_id in range(n_threads):
            thread = threading.Thread(target=make_predictions, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        concurrent_time = time.time() - start_time

        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.extend(results_queue.get())

        print("\nConcurrent Prediction Test:")
        print(f"Time for {n_threads} threads: {concurrent_time:.4f}s")
        print(f"Total predictions: {len(all_results)}")

        # Verify we got all expected results
        assert len(all_results) == n_threads * len(test_rows)

        # Verify consistency - all threads should get same results for same inputs
        predictions_by_input = {}
        for thread_id, input_idx, prediction in all_results:
            if input_idx not in predictions_by_input:
                predictions_by_input[input_idx] = []
            predictions_by_input[input_idx].append(prediction)

        for input_idx, predictions in predictions_by_input.items():
            assert len(set(predictions)) == 1, (
                f"Inconsistent predictions for input {input_idx}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output
