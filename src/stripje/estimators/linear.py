"""
Linear model estimators handlers for single-row inference.
"""

from collections.abc import Sequence
from typing import Any, Callable, Union

import numpy as np
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)

from ..registry import register_step_handler

__all__ = [
    "handle_logistic_regression",
    "handle_linear_regression",
    "handle_ridge",
    "handle_ridge_classifier",
    "handle_lasso",
    "handle_elastic_net",
    "handle_sgd_classifier",
    "handle_sgd_regressor",
]


@register_step_handler(LogisticRegression)
def handle_logistic_regression(
    step: LogisticRegression,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle LogisticRegression for single-row input."""
    coef = step.coef_
    intercept = step.intercept_
    classes = step.classes_

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        if len(classes) == 2:
            # Binary classification
            score = sum(coef[0][i] * x[i] for i in range(len(x))) + intercept[0]
            prob = 1 / (1 + np.exp(-score))
            return classes[1] if prob > 0.5 else classes[0]
        else:
            # Multi-class classification
            scores = []
            for class_idx in range(len(classes)):
                score = (
                    sum(coef[class_idx][i] * x[i] for i in range(len(x)))
                    + intercept[class_idx]
                )
                scores.append(score)
            return classes[scores.index(max(scores))]

    return predict_one


@register_step_handler(LinearRegression)
def handle_linear_regression(
    step: LinearRegression,
) -> Callable[[Sequence[Union[float, int]]], float]:
    """Handle LinearRegression for single-row input."""
    coef = step.coef_
    intercept = step.intercept_

    def predict_one(x: Sequence[Union[float, int]]) -> float:
        return float(sum(coef[i] * x[i] for i in range(len(x))) + intercept)

    return predict_one


@register_step_handler(Ridge)
def handle_ridge(step: Ridge) -> Callable[[Sequence[Union[float, int]]], float]:
    """Handle Ridge for single-row input."""
    coef = step.coef_
    intercept = step.intercept_

    def predict_one(x: Sequence[Union[float, int]]) -> float:
        return float(sum(coef[i] * x[i] for i in range(len(x))) + intercept)

    return predict_one


@register_step_handler(RidgeClassifier)
def handle_ridge_classifier(
    step: RidgeClassifier,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle RidgeClassifier for single-row input."""
    coef = step.coef_
    intercept = step.intercept_
    classes = step.classes_

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        if len(classes) == 2:
            # Binary classification - coef is 1D
            if coef.ndim == 1:
                score = sum(coef[i] * x[i] for i in range(len(x))) + intercept[0]
            else:
                score = sum(coef[0][i] * x[i] for i in range(len(x))) + intercept[0]
            return classes[1] if score > 0 else classes[0]
        else:
            # Multi-class classification - coef is 2D
            scores = []
            for class_idx in range(len(classes)):
                score = (
                    sum(coef[class_idx][i] * x[i] for i in range(len(x)))
                    + intercept[class_idx]
                )
                scores.append(score)
            return classes[scores.index(max(scores))]

    return predict_one


@register_step_handler(Lasso)
def handle_lasso(step: Lasso) -> Callable[[Sequence[Union[float, int]]], float]:
    """Handle Lasso for single-row input."""
    coef = step.coef_
    intercept = step.intercept_

    def predict_one(x: Sequence[Union[float, int]]) -> float:
        return float(sum(coef[i] * x[i] for i in range(len(x))) + intercept)

    return predict_one


@register_step_handler(ElasticNet)
def handle_elastic_net(
    step: ElasticNet,
) -> Callable[[Sequence[Union[float, int]]], float]:
    """Handle ElasticNet for single-row input."""
    coef = step.coef_
    intercept = step.intercept_

    def predict_one(x: Sequence[Union[float, int]]) -> float:
        return float(sum(coef[i] * x[i] for i in range(len(x))) + intercept)

    return predict_one


@register_step_handler(SGDClassifier)
def handle_sgd_classifier(
    step: SGDClassifier,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle SGDClassifier for single-row input."""
    coef = step.coef_
    intercept = step.intercept_
    classes = step.classes_

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        if len(classes) == 2:
            score = sum(coef[0][i] * x[i] for i in range(len(x))) + intercept[0]
            return classes[1] if score > 0 else classes[0]
        else:
            scores = []
            for class_idx in range(len(classes)):
                score = (
                    sum(coef[class_idx][i] * x[i] for i in range(len(x)))
                    + intercept[class_idx]
                )
                scores.append(score)
            return classes[scores.index(max(scores))]

    return predict_one


@register_step_handler(SGDRegressor)
def handle_sgd_regressor(
    step: SGDRegressor,
) -> Callable[[Sequence[Union[float, int]]], float]:
    """Handle SGDRegressor for single-row input."""
    coef = step.coef_
    intercept = step.intercept_

    def predict_one(x: Sequence[Union[float, int]]) -> float:
        return float(sum(coef[i] * x[i] for i in range(len(x))) + intercept)

    return predict_one
