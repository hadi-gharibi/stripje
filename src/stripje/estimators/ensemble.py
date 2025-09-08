"""
Ensemble estimators handlers for single-row inference.
"""

from collections.abc import Sequence
from typing import Any, Callable, Union

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from ..registry import register_step_handler

__all__ = [
    "handle_random_forest_classifier",
    "handle_random_forest_regressor",
    "handle_gradient_boosting_classifier",
    "handle_gradient_boosting_regressor",
]


@register_step_handler(RandomForestClassifier)
def handle_random_forest_classifier(
    step: RandomForestClassifier,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle RandomForestClassifier for single-row input."""
    estimators = step.estimators_
    classes = step.classes_

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        # Accumulate probabilities across trees (same as sklearn)
        class_probs = np.zeros(len(classes))

        for tree in estimators:
            # Simple tree traversal
            node = 0
            while tree.tree_.children_left[node] != -1:  # Not a leaf
                if x[tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                    node = tree.tree_.children_left[node]
                else:
                    node = tree.tree_.children_right[node]

            # Get class probabilities from leaf
            class_counts = tree.tree_.value[node][0]
            class_probs += class_counts / np.sum(
                class_counts
            )  # Normalize to probabilities

        # Average probabilities across trees and get class with highest probability
        avg_probs = class_probs / len(estimators)
        predicted_class_idx = np.argmax(avg_probs)

        return classes[predicted_class_idx]

    return predict_one


@register_step_handler(RandomForestRegressor)
def handle_random_forest_regressor(
    step: RandomForestRegressor,
) -> Callable[[Sequence[Union[float, int]]], float]:
    """Handle RandomForestRegressor for single-row input."""
    estimators = step.estimators_

    def predict_one(x: Sequence[Union[float, int]]) -> float:
        predictions = []
        for tree in estimators:
            # Simple tree traversal
            node = 0
            while tree.tree_.children_left[node] != -1:  # Not a leaf
                if x[tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                    node = tree.tree_.children_left[node]
                else:
                    node = tree.tree_.children_right[node]

            # Get prediction from leaf
            predictions.append(tree.tree_.value[node][0][0])

        return sum(predictions) / len(predictions)

    return predict_one


@register_step_handler(GradientBoostingClassifier)
def handle_gradient_boosting_classifier(
    step: GradientBoostingClassifier,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle GradientBoostingClassifier for single-row input."""
    estimators = step.estimators_
    classes = step.classes_
    init_pred = step._raw_predict_init(np.array([[0] * step.n_features_in_]))[0]
    learning_rate = step.learning_rate

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        if len(classes) == 2:
            # Binary classification
            score = init_pred[0]
            for stage_estimators in estimators:
                tree = stage_estimators[0]
                node = 0
                while tree.tree_.children_left[node] != -1:
                    if x[tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                        node = tree.tree_.children_left[node]
                    else:
                        node = tree.tree_.children_right[node]
                score += learning_rate * tree.tree_.value[node][0][0]

            prob = 1 / (1 + np.exp(-score))
            return classes[1] if prob > 0.5 else classes[0]
        else:
            # Multi-class classification
            scores = list(init_pred)
            for stage_estimators in estimators:
                for class_idx, tree in enumerate(stage_estimators):
                    node = 0
                    while tree.tree_.children_left[node] != -1:
                        if x[tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                            node = tree.tree_.children_left[node]
                        else:
                            node = tree.tree_.children_right[node]
                    scores[class_idx] += learning_rate * tree.tree_.value[node][0][0]

            return classes[scores.index(max(scores))]

    return predict_one


@register_step_handler(GradientBoostingRegressor)
def handle_gradient_boosting_regressor(
    step: GradientBoostingRegressor,
) -> Callable[[Sequence[Union[float, int]]], float]:
    """Handle GradientBoostingRegressor for single-row input."""
    estimators = step.estimators_
    init_pred = step._raw_predict_init(np.array([[0] * step.n_features_in_]))[0]
    learning_rate = step.learning_rate

    def predict_one(x: Sequence[Union[float, int]]) -> float:
        score = (
            init_pred[0] if hasattr(init_pred, "__len__") else init_pred
        )  # Handle array vs scalar
        for stage_estimators in estimators:
            tree = stage_estimators[0]
            node = 0
            while tree.tree_.children_left[node] != -1:
                if x[tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                    node = tree.tree_.children_left[node]
                else:
                    node = tree.tree_.children_right[node]
            score += learning_rate * tree.tree_.value[node][0][0]

        return float(score)

    return predict_one
