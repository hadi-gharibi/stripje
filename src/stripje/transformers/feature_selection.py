"""
Feature selection transformers handlers for single-row inference.
"""

from collections.abc import Callable, Sequence

from sklearn.feature_selection import (
    RFE,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
)

from ..registry import register_step_handler


@register_step_handler(SelectKBest)
def handle_select_k_best(
    step: SelectKBest,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle SelectKBest for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


@register_step_handler(SelectPercentile)
def handle_select_percentile(
    step: SelectPercentile,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle SelectPercentile for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


@register_step_handler(VarianceThreshold)
def handle_variance_threshold(
    step: VarianceThreshold,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle VarianceThreshold for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


@register_step_handler(SelectFromModel)
def handle_select_from_model(
    step: SelectFromModel,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle SelectFromModel for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


@register_step_handler(RFE)
def handle_rfe(
    step: RFE,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle RFE for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


@register_step_handler(SelectFdr)
def handle_select_fdr(
    step: SelectFdr,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle SelectFdr for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


@register_step_handler(SelectFpr)
def handle_select_fpr(
    step: SelectFpr,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle SelectFpr for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


@register_step_handler(SelectFwe)
def handle_select_fwe(
    step: SelectFwe,
) -> Callable[[Sequence[float | int]], list[float | int]]:
    """Handle SelectFwe for single-row input."""
    mask = step.get_support()

    def transform_one(x: Sequence[float | int]) -> list[float | int]:
        return [x[i] for i, selected in enumerate(mask) if selected]

    return transform_one


__all__ = [
    "handle_select_k_best",
    "handle_select_percentile",
    "handle_variance_threshold",
    "handle_select_from_model",
    "handle_rfe",
    "handle_select_fdr",
    "handle_select_fpr",
    "handle_select_fwe",
]
