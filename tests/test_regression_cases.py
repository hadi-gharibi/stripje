"""
Additional tests for specific transformer edge cases to prevent regression.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer

from stripje import compile_pipeline


class TestTransformerRegressionCases:
    """Test specific regression cases that were found during benchmarking."""
    
    @pytest.fixture
    def benchmark_data(self):
        """Generate the same data used in benchmarks."""
        np.random.seed(42)
        X, _ = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            random_state=42,
        )
        return X
    
    def test_quantile_transformer_normal_accuracy(self, benchmark_data):
        """Test QuantileTransformer with normal distribution for accuracy."""
        X = benchmark_data
        
        # Create transformer with exact benchmark settings
        transformer = QuantileTransformer(output_distribution="normal", n_quantiles=100)
        transformer.fit(X)
        
        # Create pipeline and compile
        pipeline = Pipeline([("component", transformer)])
        fast_func = compile_pipeline(pipeline)
        
        # Test multiple samples to ensure consistency
        for i in range(10):
            test_row = X[i].tolist()
            
            original_result = pipeline.transform([test_row])[0]
            fast_result = fast_func(test_row)
            
            # Use the same tolerance as the benchmark
            assert np.allclose(
                original_result, fast_result, rtol=1e-3, atol=1e-3
            ), f"QuantileTransformer normal mismatch for sample {i}"
    
    def test_quantile_transformer_uniform_accuracy(self, benchmark_data):
        """Test QuantileTransformer with uniform distribution for accuracy."""
        X = benchmark_data
        
        # Create transformer with exact benchmark settings
        transformer = QuantileTransformer(output_distribution="uniform", n_quantiles=100)
        transformer.fit(X)
        
        # Create pipeline and compile
        pipeline = Pipeline([("component", transformer)])
        fast_func = compile_pipeline(pipeline)
        
        # Test multiple samples to ensure consistency
        for i in range(10):
            test_row = X[i].tolist()
            
            original_result = pipeline.transform([test_row])[0]
            fast_result = fast_func(test_row)
            
            # Uniform should be more precise
            assert np.allclose(
                original_result, fast_result, rtol=1e-10, atol=1e-10
            ), f"QuantileTransformer uniform mismatch for sample {i}"
    
    def test_kbins_discretizer_onehot_sparse_handling(self, benchmark_data):
        """Test KBinsDiscretizer with onehot encoding handles sparse matrices correctly."""
        X = benchmark_data
        
        # Create transformer with exact benchmark settings
        transformer = KBinsDiscretizer(n_bins=5, encode="onehot")
        transformer.fit(X)
        
        # Create pipeline and compile
        pipeline = Pipeline([("component", transformer)])
        fast_func = compile_pipeline(pipeline)
        
        # Test multiple samples
        for i in range(10):
            test_row = X[i].tolist()
            
            # Get original result (sparse matrix)
            original_result_sparse = pipeline.transform([test_row])
            original_result = original_result_sparse.toarray()[0]  # Convert to dense
            
            fast_result = fast_func(test_row)
            
            assert np.allclose(
                original_result, fast_result, rtol=1e-10, atol=1e-10
            ), f"KBinsDiscretizer onehot mismatch for sample {i}"
            
            # Verify sparse matrix was handled correctly
            assert hasattr(original_result_sparse, 'toarray'), "Should return sparse matrix"
    
    def test_kbins_discretizer_ordinal_accuracy(self, benchmark_data):
        """Test KBinsDiscretizer with ordinal encoding for accuracy."""
        X = benchmark_data
        
        # Create transformer with exact benchmark settings
        transformer = KBinsDiscretizer(n_bins=5, encode="ordinal")
        transformer.fit(X)
        
        # Create pipeline and compile
        pipeline = Pipeline([("component", transformer)])
        fast_func = compile_pipeline(pipeline)
        
        # Test multiple samples
        for i in range(10):
            test_row = X[i].tolist()
            
            original_result = pipeline.transform([test_row])[0]
            fast_result = fast_func(test_row)
            
            assert np.allclose(
                original_result, fast_result, rtol=1e-10, atol=1e-10
            ), f"KBinsDiscretizer ordinal mismatch for sample {i}"
    
    def test_inverse_normal_cdf_edge_cases(self):
        """Test inverse normal CDF implementation with edge cases."""
        from stripje.transformers.preprocessing import handle_quantile_transformer
        from sklearn.preprocessing import QuantileTransformer
        
        # Create a simple transformer for testing
        X = np.random.randn(100, 1)
        transformer = QuantileTransformer(output_distribution="normal", n_quantiles=50)
        transformer.fit(X)
        
        fast_func = handle_quantile_transformer(transformer)
        
        # Test edge cases that were problematic
        edge_cases = [
            # Very low quantiles (tail region)
            [-3.0],  # Should map to low quantile
            [-2.0],
            [-1.0],
            # Central region
            [0.0],
            [1.0],
            # High quantiles
            [2.0],
            [3.0],
        ]
        
        for test_case in edge_cases:
            fast_result = fast_func(test_case)
            original_result = transformer.transform([test_case])[0]
            
            # Allow reasonable tolerance for approximation
            assert np.allclose(
                original_result, fast_result, rtol=1e-3, atol=1e-3
            ), f"Inverse normal CDF failed for input {test_case}"
    
    def test_benchmark_verification_logic(self, benchmark_data):
        """Test the benchmark verification logic works correctly."""
        X = benchmark_data
        
        # Test components that had issues
        transformers = [
            ("QuantileTransformer_normal", QuantileTransformer(output_distribution="normal", n_quantiles=100)),
            ("KBinsDiscretizer_onehot", KBinsDiscretizer(n_bins=5, encode="onehot")),
            ("QuantileTransformer_uniform", QuantileTransformer(output_distribution="uniform", n_quantiles=100)),
            ("KBinsDiscretizer_ordinal", KBinsDiscretizer(n_bins=5, encode="ordinal")),
        ]
        
        for name, transformer in transformers:
            transformer.fit(X)
            pipeline = Pipeline([("component", transformer)])
            fast_func = compile_pipeline(pipeline)
            
            test_row = X[0].tolist()
            
            # Simulate benchmark verification logic
            original_result_raw = pipeline.transform([test_row])
            if hasattr(original_result_raw, 'toarray'):
                original_result = original_result_raw.toarray()[0]
            else:
                original_result = original_result_raw[0]
            
            fast_result = fast_func(test_row)
            
            # Apply component-specific tolerance
            component = pipeline.named_steps.get("component") or list(pipeline.named_steps.values())[0]
            if hasattr(component, 'output_distribution') and component.output_distribution == 'normal':
                # Relaxed tolerance for normal distribution
                verification = np.allclose(original_result, fast_result, rtol=1e-3, atol=1e-3)
            else:
                # Strict tolerance for others
                verification = np.allclose(original_result, fast_result, rtol=1e-10, atol=1e-10)
            
            assert verification, f"Benchmark verification failed for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
