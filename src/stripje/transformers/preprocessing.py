"""
Preprocessing transformers handlers for single-row inference.
"""

from collections.abc import Sequence
from typing import Any, Callable, Union

import numpy as np
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from ..registry import register_step_handler


@register_step_handler(StandardScaler)
def handle_standard_scaler(
    step: StandardScaler,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle StandardScaler for single-row input."""
    mean = step.mean_
    scale = step.scale_

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        return [(val - mean[i]) / scale[i] for i, val in enumerate(x)]

    return transform_one


@register_step_handler(MinMaxScaler)
def handle_minmax_scaler(
    step: MinMaxScaler,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle MinMaxScaler for single-row input."""
    data_min = step.data_min_
    data_range = step.data_max_ - step.data_min_

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        return [(val - data_min[i]) / data_range[i] for i, val in enumerate(x)]

    return transform_one


@register_step_handler(RobustScaler)
def handle_robust_scaler(
    step: RobustScaler,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle RobustScaler for single-row input."""
    center = step.center_
    scale = step.scale_

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        return [(val - center[i]) / scale[i] for i, val in enumerate(x)]

    return transform_one


@register_step_handler(MaxAbsScaler)
def handle_maxabs_scaler(
    step: MaxAbsScaler,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle MaxAbsScaler for single-row input."""
    scale = step.scale_

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        return [val / scale[i] for i, val in enumerate(x)]

    return transform_one


@register_step_handler(Normalizer)
def handle_normalizer(
    step: Normalizer,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle Normalizer for single-row input."""
    norm = step.norm

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        if norm == "l2":
            norm_val = sum(val**2 for val in x) ** 0.5
        elif norm == "l1":
            norm_val = sum(abs(val) for val in x)
        elif norm == "max":
            norm_val = max(abs(val) for val in x)
        else:
            norm_val = 1.0

        if norm_val == 0:
            return [0.0] * len(x)
        return [val / norm_val for val in x]

    return transform_one


@register_step_handler(OneHotEncoder)
def handle_onehot_encoder(
    step: OneHotEncoder,
) -> Callable[[Sequence[Any]], list[float]]:
    """Handle OneHotEncoder for single-row input."""
    categories = step.categories_
    drop_idx = step.drop_idx_

    def transform_one(x: Sequence[Any]) -> list[float]:
        result = []
        for i, val in enumerate(x):
            category_list = categories[i]
            # Create one-hot vector for this feature
            onehot = [0.0] * len(category_list)
            try:
                idx = list(category_list).index(val)
                onehot[idx] = 1.0
            except ValueError:
                # Unknown category, all zeros
                pass

            # Handle drop_idx if specified
            if drop_idx is not None and i < len(drop_idx) and drop_idx[i] is not None:
                onehot.pop(drop_idx[i])

            result.extend(onehot)
        return result

    return transform_one


@register_step_handler(OrdinalEncoder)
def handle_ordinal_encoder(
    step: OrdinalEncoder,
) -> Callable[[Sequence[Any]], list[float]]:
    """Handle OrdinalEncoder for single-row input."""
    categories = step.categories_

    def transform_one(x: Sequence[Any]) -> list[float]:
        result = []
        for i, val in enumerate(x):
            try:
                encoded_val = list(categories[i]).index(val)
            except ValueError:
                # Unknown category, use -1 or handle based on unknown_value
                encoded_val = -1
            result.append(float(encoded_val))
        return result

    return transform_one


@register_step_handler(LabelEncoder)
def handle_label_encoder(
    step: LabelEncoder,
) -> Callable[[Union[Any, Sequence[Any]]], list[float]]:
    """Handle LabelEncoder for single-row input."""
    classes = step.classes_

    def transform_one(x: Union[Any, Sequence[Any]]) -> list[float]:
        # x should be a single value for LabelEncoder
        val = x[0] if isinstance(x, (list, tuple)) else x
        try:
            return [float(list(classes).index(val))]
        except ValueError:
            # Unknown class
            return [-1.0]

    return transform_one


@register_step_handler(LabelBinarizer)
def handle_label_binarizer(
    step: LabelBinarizer,
) -> Callable[[Union[Any, Sequence[Any]]], list[float]]:
    """Handle LabelBinarizer for single-row input."""
    classes = step.classes_

    def transform_one(x: Union[Any, Sequence[Any]]) -> list[float]:
        val = x[0] if isinstance(x, (list, tuple)) else x
        result = [0.0] * len(classes)
        try:
            idx = list(classes).index(val)
            result[idx] = 1.0
        except ValueError:
            # Unknown class, all zeros
            pass
        return result

    return transform_one


@register_step_handler(QuantileTransformer)
def handle_quantile_transformer(step):
    """Handle QuantileTransformer for single-row input."""
    quantiles = step.quantiles_
    output_distribution = step.output_distribution

    def transform_one(x):
        result = []
        for i, val in enumerate(x):
            # Find quantile for this value
            quantile_vals = quantiles[:, i]
            # Simple linear interpolation
            if val <= quantile_vals[0]:
                quantile = 0.0
            elif val >= quantile_vals[-1]:
                quantile = 1.0
            else:
                # Linear interpolation between quantiles
                for j in range(len(quantile_vals) - 1):
                    if quantile_vals[j] <= val <= quantile_vals[j + 1]:
                        ratio = (val - quantile_vals[j]) / (
                            quantile_vals[j + 1] - quantile_vals[j]
                        )
                        quantile = (j + ratio) / (len(quantile_vals) - 1)
                        break

            # Transform based on output distribution
            if output_distribution == "uniform":
                result.append(quantile)
            elif output_distribution == "normal":
                # Inverse normal CDF approximation
                if quantile <= 0:
                    result.append(-6.0)
                elif quantile >= 1:
                    result.append(6.0)
                else:
                    # Simple approximation of inverse normal CDF
                    from math import log, sqrt

                    if quantile < 0.5:
                        t = sqrt(-2 * log(quantile))
                        result.append(
                            -(
                                t
                                - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                                / (
                                    1
                                    + 1.432788 * t
                                    + 0.189269 * t * t
                                    + 0.001308 * t * t * t
                                )
                            )
                        )
                    else:
                        t = sqrt(-2 * log(1 - quantile))
                        result.append(
                            t
                            - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                            / (
                                1
                                + 1.432788 * t
                                + 0.189269 * t * t
                                + 0.001308 * t * t * t
                            )
                        )
        return result

    return transform_one


@register_step_handler(PowerTransformer)
def handle_power_transformer(step):
    """Handle PowerTransformer for single-row input."""
    lambdas = step.lambdas_
    method = step.method
    standardize = step.standardize

    # Get standardization parameters if standardize=True
    if standardize and hasattr(step, "_scaler"):
        scaler_mean = step._scaler.mean_
        scaler_scale = step._scaler.scale_
    else:
        scaler_mean = None
        scaler_scale = None

    def transform_one(x):
        result = []
        for i, val in enumerate(x):
            lmbda = lambdas[i]

            if method == "yeo-johnson":
                if val >= 0:
                    if abs(lmbda) < 1e-8:
                        transformed = np.log(val + 1)
                    else:
                        transformed = ((val + 1) ** lmbda - 1) / lmbda
                else:
                    if abs(lmbda - 2) < 1e-8:
                        transformed = -np.log(-val + 1)
                    else:
                        transformed = -((-val + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
            else:  # box-cox
                if abs(lmbda) < 1e-8:
                    transformed = np.log(val)
                else:
                    transformed = (val**lmbda - 1) / lmbda

            # Apply standardization if enabled
            if standardize and scaler_mean is not None:
                transformed = (transformed - scaler_mean[i]) / scaler_scale[i]

            result.append(transformed)
        return result

    return transform_one


@register_step_handler(Binarizer)
def handle_binarizer(
    step: Binarizer,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle Binarizer for single-row input."""
    threshold = step.threshold

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        return [1.0 if val > threshold else 0.0 for val in x]

    return transform_one


@register_step_handler(KBinsDiscretizer)
def handle_kbins_discretizer(
    step: KBinsDiscretizer,
) -> Callable[[Sequence[Union[float, int]]], list[float]]:
    """Handle KBinsDiscretizer for single-row input."""
    bin_edges = step.bin_edges_
    n_bins = step.n_bins_
    encode = step.encode

    def transform_one(x: Sequence[Union[float, int]]) -> list[float]:
        if encode == "onehot":
            result = []
            for i, val in enumerate(x):
                edges = bin_edges[i]
                # Find which bin this value falls into
                bin_idx = np.digitize(val, edges) - 1
                bin_idx = max(0, min(bin_idx, n_bins[i] - 1))

                # Create one-hot encoding
                onehot = [0.0] * n_bins[i]
                onehot[bin_idx] = 1.0
                result.extend(onehot)
            return result
        else:  # ordinal
            result = []
            for i, val in enumerate(x):
                edges = bin_edges[i]
                bin_idx = np.digitize(val, edges) - 1
                bin_idx = max(0, min(bin_idx, n_bins[i] - 1))
                result.append(float(bin_idx))
            return result

    return transform_one


# @register_step_handler(PolynomialFeatures)
# def handle_polynomial_features(step):
#     """Handle PolynomialFeatures for single-row input."""
#     powers = step.powers_

#     def transform_one(x):
#         result = []
#         for power_combination in powers:
#             # Compute the polynomial term
#             term = 1.0
#             for i, power in enumerate(power_combination):
#                 if power > 0:
#                     term *= x[i] ** power
#             result.append(term)
#         return result

#     return transform_one


@register_step_handler(TargetEncoder)
def handle_target_encoder(step):
    """Handle TargetEncoder for single-row input."""
    categories = step.categories_
    encodings = step.encodings_
    target_mean = step.target_mean_
    classes = step.classes_
    target_type = step.target_type_
    n_classes = len(classes)
    len(categories)

    # For multiclass, target_mean_ is an array. We need the mean across classes.
    if hasattr(target_mean, "__len__") and len(target_mean) > 1:
        global_mean = float(target_mean.mean())
    else:
        global_mean = (
            float(target_mean) if hasattr(target_mean, "__len__") else target_mean
        )

    def transform_one(x):
        result = []

        if target_type == "binary":
            # For binary classification, output shape = input shape
            # encodings_ has one entry per input feature
            for feature_idx, val in enumerate(x):
                val_str = str(val)
                feature_categories = categories[feature_idx]
                feature_encodings = encodings[feature_idx]

                try:
                    category_idx = list(feature_categories).index(val_str)
                    encoded_val = feature_encodings[category_idx]
                except ValueError:
                    # Unknown category, use global target mean
                    encoded_val = global_mean

                result.append(float(encoded_val))

        else:
            # For multiclass, output shape = input_features * n_classes
            # encodings_ has one entry per output feature (input_features * n_classes)
            for feature_idx, val in enumerate(x):
                val_str = str(val)
                feature_categories = categories[feature_idx]

                try:
                    category_idx = list(feature_categories).index(val_str)
                    # For multiclass, we get the one-hot encoding for this category
                    # The encodings are organized as: [feature0_class0, feature0_class1, ..., feature1_class0, ...]
                    for class_idx in range(n_classes):
                        encoding_idx = feature_idx * n_classes + class_idx
                        encoded_val = encodings[encoding_idx][category_idx]
                        result.append(float(encoded_val))
                except ValueError:
                    # Unknown category, use global target mean for all classes
                    for class_idx in range(n_classes):
                        result.append(global_mean)

        return result

    return transform_one


@register_step_handler(FunctionTransformer)
def handle_function_transformer(
    step: FunctionTransformer,
) -> Callable[[Sequence[Union[float, int]]], list[Union[float, int]]]:
    """Handle FunctionTransformer for single-row input."""
    func = step.func
    kw_args = step.kw_args if step.kw_args else {}

    def transform_one(x: Sequence[Union[float, int]]) -> list[Union[float, int]]:
        # Handle passthrough case where func is None
        if func is None:
            # This is a passthrough transformer, just return the input
            return list(x)

        # Convert to numpy array for the function
        x_array = np.array([x])
        # Apply the function
        result = func(x_array, **kw_args)
        # Convert back to list format and return the first (and only) row
        if hasattr(result, "tolist"):
            return result.tolist()[0]
        elif isinstance(result, (list, tuple)):
            return list(result[0])
        else:
            return [result] if not isinstance(result, (list, tuple)) else list(result)

    return transform_one


__all__ = [
    "handle_standard_scaler",
    "handle_minmax_scaler",
    "handle_robust_scaler",
    "handle_maxabs_scaler",
    "handle_normalizer",
    "handle_onehot_encoder",
    "handle_ordinal_encoder",
    "handle_label_encoder",
    "handle_label_binarizer",
    "handle_quantile_transformer",
    "handle_power_transformer",
    "handle_binarizer",
    "handle_kbins_discretizer",
    "handle_target_encoder",
    "handle_function_transformer",
]
