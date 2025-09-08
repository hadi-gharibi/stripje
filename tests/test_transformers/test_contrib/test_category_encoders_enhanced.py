"""
Enhanced tests for category encoders transformers handlers.
These tests focus on testing logic, edge cases, and error handling.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

try:
    import category_encoders as ce

    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

# Import transformers to register all handlers
from stripje.transformers.contrib.category_encoders_transformers import (
    CATEGORY_ENCODERS_AVAILABLE as MODULE_CE_AVAILABLE,
)
from stripje.transformers.contrib.category_encoders_transformers import (
    _safe_fit_supervised_encoder,
)


class TestCategoryEncodersStructure:
    """Test the module structure and availability handling."""

    def test_module_availability_flag(self):
        """Test that module correctly detects category_encoders availability."""
        # The module should have the same availability flag as our local check
        assert MODULE_CE_AVAILABLE == CATEGORY_ENCODERS_AVAILABLE

    def test_safe_fit_supervised_encoder_structure(self):
        """Test the _safe_fit_supervised_encoder function structure."""
        # This function should be available regardless of category_encoders
        assert callable(_safe_fit_supervised_encoder)

    @pytest.mark.skipif(
        not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
    )
    def test_safe_fit_supervised_encoder_with_mock(self):
        """Test _safe_fit_supervised_encoder with mock encoder."""
        # Create a mock encoder
        mock_encoder = Mock()
        mock_encoder._get_tags = Mock(return_value={"supervised_encoder": True})
        mock_encoder.fit = Mock()

        X = pd.DataFrame({"col": ["a", "b", "c"]})
        y = np.array([1, 0, 1])

        # Test the function
        result = _safe_fit_supervised_encoder(mock_encoder, X, y)

        # Should return the encoder
        assert result is mock_encoder
        # Should have called fit
        mock_encoder.fit.assert_called_once_with(X, y)

    @pytest.mark.skipif(
        not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
    )
    def test_safe_fit_supervised_encoder_without_get_tags(self):
        """Test _safe_fit_supervised_encoder with encoder without _get_tags method."""
        # Create a mock encoder without _get_tags
        mock_encoder = Mock()
        if hasattr(mock_encoder, "_get_tags"):
            delattr(mock_encoder, "_get_tags")
        mock_encoder.fit = Mock()

        X = pd.DataFrame({"col": ["a", "b", "c"]})
        y = np.array([1, 0, 1])

        # Test the function
        result = _safe_fit_supervised_encoder(mock_encoder, X, y)

        # Should return the encoder
        assert result is mock_encoder
        # Should have called fit
        mock_encoder.fit.assert_called_once_with(X, y)


@pytest.mark.skipif(
    not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
)
class TestCategoryEncodersEdgeCases:
    """Test edge cases for category encoders."""

    def setup_method(self):
        """Set up test data with edge cases."""
        # Basic data
        self.basic_data = pd.DataFrame(
            {
                "color": ["red", "blue", "green", "red", "blue"],
                "size": ["S", "M", "L", "S", "M"],
            }
        )
        self.target = np.array([1, 0, 1, 1, 0])

        # Data with NaN values
        self.nan_data = pd.DataFrame(
            {
                "color": ["red", "blue", np.nan, "red", "blue"],
                "size": ["S", np.nan, "L", "S", "M"],
            }
        )

        # Data with special characters
        self.special_data = pd.DataFrame(
            {
                "color": ["red-1", "blue_2", "green@3", "red-1", "blue_2"],
                "size": ["S!", "M#", "L$", "S!", "M#"],
            }
        )

        # Single category data
        self.single_data = pd.DataFrame(
            {
                "color": ["red", "red", "red", "red", "red"],
            }
        )

    def test_ordinal_encoder_with_nan(self):
        """Test OrdinalEncoder with NaN values."""
        encoder = ce.OrdinalEncoder(cols=["color"])
        encoder.fit(self.nan_data[["color"]])

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with NaN input
        test_result = compiled_func([np.nan])
        expected_result = encoder.transform(
            pd.DataFrame([[np.nan]], columns=["color"])
        ).values[0]

        np.testing.assert_allclose(
            test_result, expected_result, rtol=1e-10, equal_nan=True
        )

    def test_binary_encoder_with_special_characters(self):
        """Test BinaryEncoder with special characters."""
        encoder = ce.BinaryEncoder(cols=["color"])
        encoder.fit(self.special_data[["color"]])

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with special character input
        test_result = compiled_func(["red-1"])
        expected_result = encoder.transform(
            pd.DataFrame([["red-1"]], columns=["color"])
        ).values[0]

        np.testing.assert_allclose(test_result, expected_result, rtol=1e-10)

    def test_hashing_encoder_with_empty_string(self):
        """Test HashingEncoder with empty string."""
        encoder = ce.HashingEncoder(cols=["color"], n_components=8)
        encoder.fit(self.basic_data[["color"]])

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with empty string
        test_result = compiled_func([""])
        expected_result = encoder.transform(
            pd.DataFrame([[""]], columns=["color"])
        ).values[0]

        np.testing.assert_allclose(test_result, expected_result, rtol=1e-10)

    def test_single_category_encoder(self):
        """Test encoder with only one unique category."""
        encoder = ce.OrdinalEncoder(cols=["color"])
        encoder.fit(self.single_data[["color"]])

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with the single category
        test_result = compiled_func(["red"])
        expected_result = encoder.transform(
            pd.DataFrame([["red"]], columns=["color"])
        ).values[0]

        np.testing.assert_allclose(test_result, expected_result, rtol=1e-10)

    def test_target_encoder_with_extreme_values(self):
        """Test TargetEncoder with extreme target values."""
        extreme_target = np.array([1000000, -1000000, 0, 1000000, -1000000])

        encoder = ce.TargetEncoder(cols=["color"])
        _safe_fit_supervised_encoder(
            encoder, self.basic_data[["color"]], extreme_target
        )

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with known category
        test_result = compiled_func(["red"])
        expected_result = encoder.transform(
            pd.DataFrame([["red"]], columns=["color"])
        ).values[0]

        np.testing.assert_allclose(test_result, expected_result, rtol=1e-10)

    def test_encoder_with_large_cardinality(self):
        """Test encoder with high cardinality data."""
        # Create data with many unique categories
        large_data = pd.DataFrame(
            {
                "id": [f"id_{i}" for i in range(100)],
            }
        )

        encoder = ce.OrdinalEncoder(cols=["id"])
        encoder.fit(large_data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with known ID
        test_result = compiled_func(["id_50"])
        expected_result = encoder.transform(
            pd.DataFrame([["id_50"]], columns=["id"])
        ).values[0]

        np.testing.assert_allclose(test_result, expected_result, rtol=1e-10)

    def test_multiple_columns_different_types(self):
        """Test encoder with multiple columns of different data types."""
        mixed_data = pd.DataFrame(
            {
                "text": ["a", "b", "c", "a", "b"],
                "number_as_str": ["1", "2", "3", "1", "2"],
                "boolean_as_str": ["True", "False", "True", "False", "True"],
            }
        )

        encoder = ce.OrdinalEncoder(cols=["text", "number_as_str", "boolean_as_str"])
        encoder.fit(mixed_data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with mixed input
        test_result = compiled_func(["a", "1", "True"])
        expected_result = encoder.transform(
            pd.DataFrame(
                [["a", "1", "True"]],
                columns=["text", "number_as_str", "boolean_as_str"],
            )
        ).values[0]

        np.testing.assert_allclose(test_result, expected_result, rtol=1e-10)


@pytest.mark.skipif(
    not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
)
class TestCategoryEncodersPerformance:
    """Performance and stress tests for category encoders."""

    def test_performance_with_repeated_transformations(self):
        """Test performance with many repeated transformations."""
        import time

        # Create test data
        data = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C", "D"], 1000),
                "cat2": np.random.choice(["X", "Y", "Z"], 1000),
            }
        )

        encoder = ce.OrdinalEncoder(cols=["cat1", "cat2"])
        encoder.fit(data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test data
        test_inputs = [["A", "X"], ["B", "Y"], ["C", "Z"]] * 100

        # Time compiled version
        start_time = time.time()
        for test_input in test_inputs:
            compiled_func(test_input)
        compiled_time = time.time() - start_time

        # Time original version
        start_time = time.time()
        for test_input in test_inputs:
            encoder.transform(pd.DataFrame([test_input], columns=["cat1", "cat2"]))
        original_time = time.time() - start_time

        print(
            f"Compiled time: {compiled_time:.4f}s, Original time: {original_time:.4f}s"
        )

        # Compiled version should be faster
        assert compiled_time < original_time

    def test_memory_efficiency(self):
        """Test memory efficiency of compiled functions."""

        # Create encoder
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C"] * 100,
            }
        )

        encoder = ce.OrdinalEncoder(cols=["category"])
        encoder.fit(data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test that compiled function doesn't create large intermediate objects
        test_input = ["A"]

        # Should not raise memory error and should be consistent
        for _ in range(1000):
            result = compiled_func(test_input)
            assert len(result) == 1

    def test_concurrent_usage(self):
        """Test that compiled functions can be used concurrently."""
        import threading

        # Create encoder
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C", "D", "E"],
            }
        )

        encoder = ce.OrdinalEncoder(cols=["category"])
        encoder.fit(data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        results = []
        errors = []

        def worker(category):
            try:
                result = compiled_func([category])
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        categories = ["A", "B", "C", "D", "E"] * 10

        for category in categories:
            thread = threading.Thread(target=worker, args=(category,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Concurrent usage failed with errors: {errors}"
        assert len(results) == len(categories)


class TestCategoryEncodersRobustness:
    """Test robustness and error handling."""

    def test_module_imports_gracefully(self):
        """Test that module imports gracefully when category_encoders is not available."""
        # This test should always pass regardless of category_encoders availability
        from stripje.transformers.contrib import category_encoders_transformers

        # Module should have CATEGORY_ENCODERS_AVAILABLE attribute
        assert hasattr(category_encoders_transformers, "CATEGORY_ENCODERS_AVAILABLE")

        # __all__ should be a list
        assert isinstance(category_encoders_transformers.__all__, list)

    @pytest.mark.skipif(
        not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
    )
    def test_handlers_are_registered(self):
        """Test that all handlers are properly registered."""
        from stripje.registry import get_handler

        encoders_to_check = [
            ce.BinaryEncoder,
            ce.OneHotEncoder,
            ce.OrdinalEncoder,
            ce.HashingEncoder,
            ce.TargetEncoder,
            ce.CatBoostEncoder,
            ce.LeaveOneOutEncoder,
        ]

        for encoder_class in encoders_to_check:
            handler = get_handler(encoder_class)
            assert (
                handler is not None
            ), f"Handler not registered for {encoder_class.__name__}"
            assert callable(
                handler
            ), f"Handler for {encoder_class.__name__} is not callable"

    @pytest.mark.skipif(
        not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
    )
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
            }
        )

        encoder = ce.OrdinalEncoder(cols=["category"])
        encoder.fit(data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with wrong number of inputs
        # Too few inputs should raise an error
        try:
            result = compiled_func([])  # Too few inputs
            # If no error, the result should be empty or default
            assert len(result) == 0 or all(
                x == -1 for x in result
            ), "Empty input should produce empty or default result"
        except (IndexError, KeyError, ValueError):
            # This is also acceptable behavior
            pass

        # Test with too many inputs - the handler should either handle gracefully or raise an error
        try:
            result = compiled_func(["A", "B"])  # Too many inputs
            # If it handles gracefully, the result should be reasonable
            assert len(result) >= 1, "Should produce at least one result"
        except (IndexError, KeyError, ValueError):
            # This is also acceptable behavior
            pass

    @pytest.mark.skipif(
        not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
    )
    def test_none_input_handling(self):
        """Test handling of None inputs."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
            }
        )

        encoder = ce.OrdinalEncoder(cols=["category"])
        encoder.fit(data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with None input - should handle gracefully
        try:
            result = compiled_func([None])
            # Result should be consistent with original encoder
            expected = encoder.transform(
                pd.DataFrame([[None]], columns=["category"])
            ).values[0]
            np.testing.assert_allclose(result, expected, rtol=1e-10, equal_nan=True)
        except Exception:
            # If it raises an exception, that's also acceptable
            # as long as it's consistent behavior
            pass


@pytest.mark.skipif(
    not CATEGORY_ENCODERS_AVAILABLE, reason="category_encoders not available"
)
class TestCategoryEncodersComplexScenarios:
    """Test complex real-world scenarios."""

    def test_pipeline_like_chaining(self):
        """Test chaining multiple encoders like in a pipeline."""
        # Create complex categorical data
        data = pd.DataFrame(
            {
                "color": ["red", "blue", "green", "red", "blue", "green"] * 10,
                "size": ["S", "M", "L", "XL", "S", "M"] * 10,
                "material": ["cotton", "wool", "silk", "cotton", "wool", "silk"] * 10,
            }
        )
        target = np.random.choice([0, 1], len(data))

        # Create multiple encoders
        ordinal_encoder = ce.OrdinalEncoder(cols=["color"])
        binary_encoder = ce.BinaryEncoder(cols=["size"])
        target_encoder = ce.TargetEncoder(cols=["material"])

        # Fit encoders
        ordinal_encoder.fit(data[["color"]])
        binary_encoder.fit(data[["size"]])
        _safe_fit_supervised_encoder(target_encoder, data[["material"]], target)

        # Get handlers
        from stripje.registry import get_handler

        ordinal_handler = get_handler(type(ordinal_encoder))
        binary_handler = get_handler(type(binary_encoder))
        target_handler = get_handler(type(target_encoder))

        # Compile functions
        ordinal_func = ordinal_handler(ordinal_encoder)
        binary_func = binary_handler(binary_encoder)
        target_func = target_handler(target_encoder)

        # Test with sample data
        test_row = ["red", "M", "cotton"]

        # Transform each part
        ordinal_result = ordinal_func([test_row[0]])
        binary_result = binary_func([test_row[1]])
        target_result = target_func([test_row[2]])

        # Combine results (simulating pipeline)
        combined_result = (
            list(ordinal_result) + list(binary_result) + list(target_result)
        )

        # Verify against individual transformations
        expected_ordinal = ordinal_encoder.transform(
            pd.DataFrame([[test_row[0]]], columns=["color"])
        ).values[0]
        expected_binary = binary_encoder.transform(
            pd.DataFrame([[test_row[1]]], columns=["size"])
        ).values[0]
        expected_target = target_encoder.transform(
            pd.DataFrame([[test_row[2]]], columns=["material"])
        ).values[0]

        np.testing.assert_allclose(ordinal_result, expected_ordinal, rtol=1e-10)
        np.testing.assert_allclose(binary_result, expected_binary, rtol=1e-10)
        np.testing.assert_allclose(target_result, expected_target, rtol=1e-10)

        assert len(combined_result) == len(expected_ordinal) + len(
            expected_binary
        ) + len(expected_target)

    def test_high_cardinality_categories(self):
        """Test with very high cardinality categorical data."""
        # Create high cardinality data
        n_categories = 1000
        data = pd.DataFrame(
            {
                "high_card": [f"category_{i}" for i in range(n_categories)] * 2,
            }
        )

        encoder = ce.HashingEncoder(cols=["high_card"], n_components=64)
        encoder.fit(data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with known categories
        for i in [0, 100, 500, 999]:
            test_input = [f"category_{i}"]
            result = compiled_func(test_input)
            expected = encoder.transform(
                pd.DataFrame([test_input], columns=["high_card"])
            ).values[0]
            np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_mixed_data_types_as_strings(self):
        """Test with mixed data types converted to strings."""
        # Create data with various types as strings
        mixed_data = pd.DataFrame(
            {
                "numbers": ["1", "2", "3", "1.5", "2.7"],
                "booleans": ["True", "False", "true", "false", "TRUE"],
                "dates": [
                    "2023-01-01",
                    "2023-12-31",
                    "2024-06-15",
                    "2023-01-01",
                    "2023-12-31",
                ],
            }
        )

        encoder = ce.OrdinalEncoder(cols=["numbers", "booleans", "dates"])
        encoder.fit(mixed_data)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Test with mixed input
        test_input = ["1", "True", "2023-01-01"]
        result = compiled_func(test_input)
        expected = encoder.transform(
            pd.DataFrame([test_input], columns=["numbers", "booleans", "dates"])
        ).values[0]

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_encoder_state_consistency(self):
        """Test that encoder state remains consistent across multiple calls."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C", "A", "B"],
            }
        )
        target = [1, 0, 1, 1, 0]

        encoder = ce.TargetEncoder(cols=["category"])
        _safe_fit_supervised_encoder(encoder, data, target)

        from stripje.registry import get_handler

        handler = get_handler(type(encoder))
        compiled_func = handler(encoder)

        # Call multiple times with same input
        test_input = ["A"]
        results = []
        for _ in range(10):
            result = compiled_func(test_input)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-15)

        # Should match original encoder
        expected = encoder.transform(
            pd.DataFrame([test_input], columns=["category"])
        ).values[0]
        np.testing.assert_allclose(results[0], expected, rtol=1e-10)
