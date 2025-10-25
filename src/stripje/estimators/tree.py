"""
Tree-based estimators handlers for single-row inference.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..registry import register_step_handler


@register_step_handler(DecisionTreeClassifier)
def handle_decision_tree_classifier(
    step: DecisionTreeClassifier,
) -> Callable[[Sequence[float | int]], Any]:
    """Handle DecisionTreeClassifier for single-row input."""
    tree = step.tree_
    classes = step.classes_

    def predict_one(x: Sequence[float | int]) -> Any:
        node = 0
        while tree.children_left[node] != -1:  # Not a leaf
            if x[tree.feature[node]] <= tree.threshold[node]:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]

        # Get class prediction from leaf
        class_counts = tree.value[node][0]
        predicted_class = np.argmax(class_counts)
        return classes[predicted_class]

    return predict_one


@register_step_handler(DecisionTreeRegressor)
def handle_decision_tree_regressor(
    step: DecisionTreeRegressor,
) -> Callable[[Sequence[float | int]], float]:
    """Handle DecisionTreeRegressor for single-row input."""
    tree = step.tree_

    def predict_one(x: Sequence[float | int]) -> float:
        node = 0
        while tree.children_left[node] != -1:  # Not a leaf
            if x[tree.feature[node]] <= tree.threshold[node]:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]

        # Get prediction from leaf
        return float(tree.value[node][0][0])

    return predict_one


__all__ = ["handle_decision_tree_classifier", "handle_decision_tree_regressor"]
