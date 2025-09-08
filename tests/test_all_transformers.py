"""
Comprehensive tests for all transformers to verify output matches sklearn.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from stripje import compile_pipeline


def test_preprocessing_transformers():
    """Test all preprocessing transformers."""
    print("Testing preprocessing transformers...")

    # Generate test data
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42
    )

    # Test StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)

    from stripje.transformers.preprocessing import handle_standard_scaler

    fast_scaler = handle_standard_scaler(scaler)

    test_row = X[0].tolist()
    original_result = scaler.transform([test_row])[0]
    fast_result = fast_scaler(test_row)

    assert np.allclose(original_result, fast_result), "StandardScaler mismatch"
    print("✓ StandardScaler")

    # Test MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)

    from stripje.transformers.preprocessing import handle_minmax_scaler

    fast_scaler = handle_minmax_scaler(scaler)

    original_result = scaler.transform([test_row])[0]
    fast_result = fast_scaler(test_row)

    assert np.allclose(original_result, fast_result), "MinMaxScaler mismatch"
    print("✓ MinMaxScaler")

    # Test RobustScaler
    scaler = RobustScaler()
    scaler.fit(X)

    from stripje.transformers.preprocessing import handle_robust_scaler

    fast_scaler = handle_robust_scaler(scaler)

    original_result = scaler.transform([test_row])[0]
    fast_result = fast_scaler(test_row)

    assert np.allclose(original_result, fast_result), "RobustScaler mismatch"
    print("✓ RobustScaler")

    # Test Normalizer
    normalizer = Normalizer()
    normalizer.fit(X)

    from stripje.transformers.preprocessing import handle_normalizer

    fast_normalizer = handle_normalizer(normalizer)

    original_result = normalizer.transform([test_row])[0]
    fast_result = fast_normalizer(test_row)

    assert np.allclose(original_result, fast_result), "Normalizer mismatch"
    print("✓ Normalizer")

    # Test OneHotEncoder with categorical data
    cat_data = np.array([["A"], ["B"], ["C"], ["A"], ["B"]])
    encoder = OneHotEncoder()
    encoder.fit(cat_data)

    from stripje.transformers.preprocessing import handle_onehot_encoder

    fast_encoder = handle_onehot_encoder(encoder)

    test_cat = ["A"]
    original_result = encoder.transform([test_cat]).toarray()[0]
    fast_result = fast_encoder(test_cat)

    assert np.allclose(original_result, fast_result), "OneHotEncoder mismatch"
    print("✓ OneHotEncoder")

    print("All preprocessing transformers passed!")


def test_feature_selection_transformers():
    """Test feature selection transformers."""
    print("\nTesting feature selection transformers...")

    # Generate test data
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, n_redundant=2, random_state=42
    )

    # Test SelectKBest
    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)

    from stripje.transformers.feature_selection import handle_select_k_best

    fast_selector = handle_select_k_best(selector)

    test_row = X[0].tolist()
    original_result = selector.transform([test_row])[0]
    fast_result = fast_selector(test_row)

    assert np.allclose(original_result, fast_result), "SelectKBest mismatch"
    print("✓ SelectKBest")

    # Test VarianceThreshold
    selector = VarianceThreshold(threshold=0.1)
    selector.fit(X)

    from stripje.transformers.feature_selection import handle_variance_threshold

    fast_selector = handle_variance_threshold(selector)

    original_result = selector.transform([test_row])[0]
    fast_result = fast_selector(test_row)

    assert np.allclose(original_result, fast_result), "VarianceThreshold mismatch"
    print("✓ VarianceThreshold")

    print("All feature selection transformers passed!")


def test_decomposition_transformers():
    """Test decomposition transformers."""
    print("\nTesting decomposition transformers...")

    # Generate test data
    X, _ = make_classification(
        n_samples=100, n_features=10, n_informative=8, random_state=42
    )

    # Test PCA
    pca = PCA(n_components=5)
    pca.fit(X)

    from stripje.transformers.decomposition import handle_pca

    fast_pca = handle_pca(pca)

    test_row = X[0].tolist()
    original_result = pca.transform([test_row])[0]
    fast_result = fast_pca(test_row)

    assert np.allclose(original_result, fast_result), "PCA mismatch"
    print("✓ PCA")

    # Test TruncatedSVD
    svd = TruncatedSVD(n_components=5)
    svd.fit(X)

    from stripje.transformers.decomposition import handle_truncated_svd

    fast_svd = handle_truncated_svd(svd)

    original_result = svd.transform([test_row])[0]
    fast_result = fast_svd(test_row)

    assert np.allclose(original_result, fast_result), "TruncatedSVD mismatch"
    print("✓ TruncatedSVD")

    print("All decomposition transformers passed!")


def test_speed_comparison():
    """Test speed improvement for different transformers."""
    print("\nTesting speed improvements...")

    import time

    from sklearn.pipeline import Pipeline

    # Generate larger test data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, random_state=42
    )

    transformers_to_test = [
        ("StandardScaler", StandardScaler()),
        ("MinMaxScaler", MinMaxScaler()),
        ("PCA", PCA(n_components=10)),
        ("SelectKBest", SelectKBest(k=10)),
    ]

    for name, transformer in transformers_to_test:
        # Fit transformer
        transformer.fit(X, y if hasattr(transformer, "score_func") else None)

        # Create pipeline
        pipeline = Pipeline([("transformer", transformer)])
        pipeline.fit(X, y if hasattr(transformer, "score_func") else None)

        # Compile pipeline
        try:
            fast_transform = compile_pipeline(pipeline)

            # Test single row
            test_row = X[0].tolist()

            # Benchmark original
            n_iterations = 1000
            start_time = time.time()
            for _ in range(n_iterations):
                pipeline.transform([test_row])
            original_time = time.time() - start_time

            # Benchmark fast
            start_time = time.time()
            for _ in range(n_iterations):
                fast_transform(test_row)
            fast_time = time.time() - start_time

            speedup = original_time / fast_time
            print(
                f"✓ {name}: {speedup:.1f}x speedup ({original_time:.4f}s vs {fast_time:.4f}s)"
            )

        except Exception as e:
            print(f"✗ {name}: {e}")


if __name__ == "__main__":
    test_preprocessing_transformers()
    test_feature_selection_transformers()
    test_decomposition_transformers()
    test_speed_comparison()
    print("\n🎉 All tests completed!")
