"""
Naive Bayes estimators handlers for single-row inference.
"""

from collections.abc import Sequence
from typing import Any, Callable, Union

import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from ..registry import register_step_handler

__all__ = ["handle_gaussian_nb", "handle_multinomial_nb", "handle_bernoulli_nb"]


@register_step_handler(GaussianNB)
def handle_gaussian_nb(
    step: GaussianNB,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle GaussianNB for single-row input."""
    theta = step.theta_
    var = step.var_  # Changed from sigma_ to var_
    class_prior = step.class_prior_
    classes = step.classes_

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        log_probs = []
        for class_idx in range(len(classes)):
            log_prob = np.log(class_prior[class_idx])
            for feature_idx in range(len(x)):
                mean = theta[class_idx][feature_idx]
                variance = var[class_idx][feature_idx]
                # Gaussian probability density
                log_prob += (
                    -0.5 * np.log(2 * np.pi * variance)
                    - 0.5 * ((x[feature_idx] - mean) ** 2) / variance
                )
            log_probs.append(log_prob)

        return classes[log_probs.index(max(log_probs))]

    return predict_one


@register_step_handler(MultinomialNB)
def handle_multinomial_nb(
    step: MultinomialNB,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle MultinomialNB for single-row input."""
    feature_log_prob = step.feature_log_prob_
    class_log_prior = step.class_log_prior_
    classes = step.classes_

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        log_probs = []
        for class_idx in range(len(classes)):
            log_prob = class_log_prior[class_idx]
            for feature_idx in range(len(x)):
                log_prob += x[feature_idx] * feature_log_prob[class_idx][feature_idx]
            log_probs.append(log_prob)

        return classes[log_probs.index(max(log_probs))]

    return predict_one


@register_step_handler(BernoulliNB)
def handle_bernoulli_nb(
    step: BernoulliNB,
) -> Callable[[Sequence[Union[float, int]]], Any]:
    """Handle BernoulliNB for single-row input."""
    feature_log_prob = step.feature_log_prob_
    class_log_prior = step.class_log_prior_
    classes = step.classes_

    def predict_one(x: Sequence[Union[float, int]]) -> Any:
        log_probs = []
        for class_idx in range(len(classes)):
            log_prob = class_log_prior[class_idx]
            for feature_idx in range(len(x)):
                if x[feature_idx]:
                    log_prob += feature_log_prob[class_idx][feature_idx]
                else:
                    log_prob += np.log(
                        1 - np.exp(feature_log_prob[class_idx][feature_idx])
                    )
            log_probs.append(log_prob)

        return classes[log_probs.index(max(log_probs))]

    return predict_one
