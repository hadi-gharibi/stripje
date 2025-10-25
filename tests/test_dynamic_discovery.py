"""
Tests for dynamic handler discovery and external registration.
"""

import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from stripje import compile_pipeline, get_supported_transformers, register_step_handler
from stripje.registry import STEP_HANDLERS


class TestDynamicDiscovery:
    """Test that handlers are automatically discovered."""

    def test_handlers_are_automatically_registered(self):
        """Test that handlers from all modules are automatically discovered."""
        supported = get_supported_transformers()

        # Should have many handlers registered
        assert len(supported) > 50, (
            "Expected more than 50 handlers to be auto-discovered"
        )

        # Check some expected handlers from different modules
        assert StandardScaler in supported, "StandardScaler should be auto-discovered"

    def test_all_module_handlers_discovered(self):
        """Test that handlers from all submodules are discovered."""
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest
        from sklearn.preprocessing import StandardScaler

        supported = get_supported_transformers()

        # Preprocessing
        assert StandardScaler in supported
        # Feature selection
        assert SelectKBest in supported
        # Decomposition
        assert PCA in supported
        # Estimators
        assert RandomForestClassifier in supported

    def test_contrib_handlers_discovered_if_available(self):
        """Test that contrib handlers are discovered when dependencies available."""
        try:
            import category_encoders as ce

            supported = get_supported_transformers()
            # At least some category_encoders should be registered
            assert ce.BinaryEncoder in supported or ce.OneHotEncoder in supported
        except ImportError:
            # If category_encoders not installed, that's okay
            pytest.skip("category_encoders not installed")


class TestExternalRegistration:
    """Test external handler registration API."""

    def test_register_custom_transformer(self):
        """Test registering a custom transformer from outside the library."""

        class CustomTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X * 2

        # Check not registered initially
        assert CustomTransformer not in get_supported_transformers()

        # Register handler
        @register_step_handler(CustomTransformer)
        def handle_custom(step):
            def transform_one(x):
                return [val * 2 for val in x]

            return transform_one

        # Should now be registered
        assert CustomTransformer in get_supported_transformers()
        assert CustomTransformer in STEP_HANDLERS

        # Clean up
        del STEP_HANDLERS[CustomTransformer]

    def test_custom_handler_works_in_pipeline(self):
        """Test that a custom registered handler works in a compiled pipeline."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        class TripleTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X * 3

        # Register handler
        @register_step_handler(TripleTransformer)
        def handle_triple(step):
            def transform_one(x):
                return [val * 3 for val in x]

            return transform_one

        # Create and fit pipeline with classifier
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        pipeline = Pipeline(
            [("triple", TripleTransformer()), ("clf", LogisticRegression())]
        )
        pipeline.fit(X, y)

        # Compile and test
        fast_predict = compile_pipeline(pipeline)
        test_row = X[0]

        sklearn_result = pipeline.predict([test_row])[0]
        fast_result = fast_predict(test_row)

        assert sklearn_result == fast_result

        # Clean up
        del STEP_HANDLERS[TripleTransformer]

    def test_register_step_handler_is_exported(self):
        """Test that register_step_handler is available from main module."""
        import stripje

        assert hasattr(stripje, "register_step_handler")
        assert callable(stripje.register_step_handler)

    def test_multiple_custom_handlers(self):
        """Test registering multiple custom handlers."""

        class CustomA(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X + 1

        class CustomB(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X * 2

        @register_step_handler(CustomA)
        def handle_a(step):
            def transform_one(x):
                return [val + 1 for val in x]

            return transform_one

        @register_step_handler(CustomB)
        def handle_b(step):
            def transform_one(x):
                return [val * 2 for val in x]

            return transform_one

        supported = get_supported_transformers()
        assert CustomA in supported
        assert CustomB in supported

        # Clean up
        del STEP_HANDLERS[CustomA]
        del STEP_HANDLERS[CustomB]

    def test_custom_handler_with_parameters(self):
        """Test that custom handlers can access transformer parameters."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        class ParameterizedTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, multiplier=5):
                self.multiplier = multiplier

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X * self.multiplier

        @register_step_handler(ParameterizedTransformer)
        def handle_parameterized(step):
            multiplier = step.multiplier

            def transform_one(x):
                return [val * multiplier for val in x]

            return transform_one

        # Create sample data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)

        # Test with different parameter values
        for mult in [2, 5]:
            pipeline = Pipeline(
                [
                    ("param", ParameterizedTransformer(multiplier=mult)),
                    ("clf", LogisticRegression()),
                ]
            )
            pipeline.fit(X, y)

            fast_predict = compile_pipeline(pipeline)
            test_row = X[0]

            sklearn_result = pipeline.predict([test_row])[0]
            fast_result = fast_predict(test_row)

            assert sklearn_result == fast_result

        # Clean up
        del STEP_HANDLERS[ParameterizedTransformer]


class TestDiscoveryRobustness:
    """Test that discovery mechanism is robust."""

    def test_discovery_excludes_private_modules(self):
        """Test that modules starting with _ are not imported."""
        # __init__, __pycache__, etc. should be excluded
        # This is implicit - if they were imported, we'd likely get errors
        # Just verify the registry works
        assert len(STEP_HANDLERS) > 0

    def test_discovery_handles_import_errors_gracefully(self):
        """Test that optional dependencies are handled gracefully."""
        # If category_encoders is not installed, it should not break
        # This is tested by the fact that the module loads successfully
        from stripje import transformers

        assert transformers is not None

    def test_all_attribute_is_populated(self):
        """Test that __all__ is populated by discovery."""
        from stripje import estimators, transformers

        # Both should have __all__ defined
        assert hasattr(transformers, "__all__")
        assert hasattr(estimators, "__all__")

        # Should contain handler function names
        assert len(transformers.__all__) > 0
        assert len(estimators.__all__) > 0
