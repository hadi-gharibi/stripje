"""
Decomposition transformers handlers for single-row inference.
"""

from collections.abc import Sequence
from typing import Callable, Union

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD

from ..registry import register_step_handler

__all__ = [
    "handle_pca",
    "handle_truncated_svd",
    "handle_fast_ica",
    "handle_factor_analysis",
]


@register_step_handler(PCA)
def handle_pca(step: PCA) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle PCA for single-row input."""
    components = step.components_
    mean = step.mean_

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        # Center the data
        centered = [val - mean[i] for i, val in enumerate(x)]
        # Apply transformation
        result = []
        for component in components:
            result.append(sum(centered[i] * component[i] for i in range(len(centered))))
        return result

    return transform_one


@register_step_handler(TruncatedSVD)
def handle_truncated_svd(
    step: TruncatedSVD,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle TruncatedSVD for single-row input."""
    components = step.components_

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        result = []
        for component in components:
            result.append(sum(x[i] * component[i] for i in range(len(x))))
        return result

    return transform_one


@register_step_handler(FastICA)
def handle_fast_ica(
    step: FastICA,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle FastICA for single-row input."""
    components = step.components_
    # FastICA only has mean_ when whiten is enabled
    mean = getattr(step, "mean_", None)

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        # Center the data if mean is available
        centered = (
            [val - mean[i] for i, val in enumerate(x)] if mean is not None else list(x)
        )

        # Apply transformation
        result = []
        for component in components:
            result.append(sum(centered[i] * component[i] for i in range(len(centered))))
        return result

    return transform_one


@register_step_handler(FactorAnalysis)
def handle_factor_analysis(
    step: FactorAnalysis,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle FactorAnalysis for single-row input."""
    components = step.components_  # (n_components, n_features)
    mean = step.mean_
    noise_variance = step.noise_variance_

    # Pre-compute transformation matrices for efficiency
    Lambda = components.T  # (n_features, n_components)
    Psi_inv = np.diag(1.0 / noise_variance)  # (n_features, n_features)
    precision = Lambda.T @ Psi_inv @ Lambda + np.eye(step.n_components)
    precision_inv = np.linalg.inv(precision)
    transform_matrix = precision_inv @ Lambda.T @ Psi_inv

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        # Center the data
        centered = [val - mean[i] for i, val in enumerate(x)]

        # Apply the factor analysis transformation
        # z = (Lambda^T @ Psi^{-1} @ Lambda + I)^{-1} @ Lambda^T @ Psi^{-1} @ (x - mu)
        result = []
        for i in range(len(transform_matrix)):
            result.append(
                sum(transform_matrix[i][j] * centered[j] for j in range(len(centered)))
            )
        return result

    return transform_one
