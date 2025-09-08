"""
Category encoders transformers handlers for single-row inference.
These are optional transformers from the category_encoders library.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import category_encoders as ce

    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False
    ce = None

from ...registry import register_step_handler


def _safe_fit_supervised_encoder(encoder: Any, X: Any, y: Any) -> Any:
    """
    Safely fit a supervised encoder by temporarily patching sklearn compatibility issues.

    This is a workaround for category_encoders compatibility issues with sklearn 1.6+.
    """
    # Save original _get_tags method if it exists
    original_get_tags = getattr(encoder, "_get_tags", None)

    # Temporarily patch _get_tags to avoid the sklearn compatibility issue
    def mock_get_tags() -> dict[str, bool]:
        return {"supervised_encoder": True}

    encoder._get_tags = mock_get_tags

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            encoder.fit(X, y)
    finally:
        # Restore original method
        if original_get_tags is not None:
            encoder._get_tags = original_get_tags
        else:
            delattr(encoder, "_get_tags")

    return encoder


if CATEGORY_ENCODERS_AVAILABLE:

    @register_step_handler(ce.BinaryEncoder)
    def handle_binary_encoder(step: Any) -> Callable[[Any], Any]:
        """Handle BinaryEncoder for single-row input."""
        # Get the mapping information
        mapping = step.mapping

        def transform_one(x: Any) -> Any:
            # Convert to dict if it's a list/array with column names
            if isinstance(x, (list, np.ndarray)):
                # Assume x is ordered according to the columns used during fit
                if hasattr(step, "cols") and step.cols:
                    x_dict = {col: x[i] for i, col in enumerate(step.cols)}
                else:
                    # Default to using integer indices as column names
                    x_dict = dict(enumerate(x))
            else:
                x_dict = x  # type: ignore[unreachable]

            result = []

            # BinaryEncoder mapping is different - it uses DataFrames
            for col_mapping in mapping:
                col = col_mapping["col"]
                mapping_df = col_mapping["mapping"]  # This is a DataFrame

                if col in x_dict:
                    value = x_dict[col]

                    # First, get the ordinal value for this categorical value
                    # BinaryEncoder internally uses OrdinalEncoder
                    ordinal_encoder = step.ordinal_encoder
                    ordinal_mapping = None

                    # Find the corresponding ordinal mapping
                    for ord_mapping in ordinal_encoder.mapping:
                        if ord_mapping["col"] == col:
                            ordinal_mapping = ord_mapping["mapping"]  # pandas Series
                            break

                    if ordinal_mapping is not None and value in ordinal_mapping.index:
                        ordinal_value = ordinal_mapping[value]

                        # Look up the binary encoding for this ordinal value
                        if ordinal_value in mapping_df.index:
                            binary_row = mapping_df.loc[ordinal_value]
                            result.extend(binary_row.tolist())
                        else:
                            # Unknown ordinal value, use default (all zeros)
                            result.extend([0] * len(mapping_df.columns))
                    else:
                        # Unknown category, use default encoding
                        result.extend([0] * len(mapping_df.columns))
                else:
                    # Column not found, use default encoding
                    result.extend([0] * len(mapping_df.columns))

            return result

        return transform_one

    @register_step_handler(ce.OneHotEncoder)
    def handle_ce_one_hot_encoder(step: Any) -> Callable[[Any], Any]:
        """Handle category_encoders OneHotEncoder for single-row input."""
        mapping = step.mapping

        def transform_one(x: Any) -> Any:
            if isinstance(x, (list, np.ndarray)):
                if hasattr(step, "cols") and step.cols:
                    x_dict = {col: x[i] for i, col in enumerate(step.cols)}
                else:
                    x_dict = dict(enumerate(x))
            else:
                x_dict = x  # type: ignore[unreachable]

            result = []

            # category_encoders OneHotEncoder also uses DataFrames
            for col_mapping in mapping:
                col = col_mapping["col"]
                mapping_df = col_mapping["mapping"]  # This is a DataFrame

                if col in x_dict:
                    value = x_dict[col]

                    # Get the ordinal value for this categorical value
                    ordinal_encoder = step.ordinal_encoder
                    ordinal_mapping = None

                    # Find the corresponding ordinal mapping
                    for ord_mapping in ordinal_encoder.mapping:
                        if ord_mapping["col"] == col:
                            ordinal_mapping = ord_mapping["mapping"]  # pandas Series
                            break

                    if ordinal_mapping is not None and value in ordinal_mapping.index:
                        ordinal_value = ordinal_mapping[value]

                        # Look up the one-hot encoding for this ordinal value
                        if ordinal_value in mapping_df.index:
                            onehot_row = mapping_df.loc[ordinal_value]
                            result.extend(onehot_row.tolist())
                        else:
                            # Unknown ordinal value, use default (all zeros)
                            result.extend([0] * len(mapping_df.columns))
                    else:
                        # Unknown category, use default encoding
                        result.extend([0] * len(mapping_df.columns))
                else:
                    # Column not found, append zeros
                    result.extend([0] * len(mapping_df.columns))

            return result

        return transform_one

    @register_step_handler(ce.OrdinalEncoder)
    def handle_ce_ordinal_encoder(step: Any) -> Callable[[Any], Any]:
        """Handle category_encoders OrdinalEncoder for single-row input."""
        mapping = step.mapping

        def transform_one(x: Any) -> Any:
            if isinstance(x, (list, np.ndarray)):
                if hasattr(step, "cols") and step.cols:
                    x_dict = {col: x[i] for i, col in enumerate(step.cols)}
                else:
                    x_dict = dict(enumerate(x))
            else:
                x_dict = x  # type: ignore[unreachable]

            result = []
            for col_mapping in mapping:
                col = col_mapping["col"]
                mapping_series = col_mapping["mapping"]  # This is a pandas Series

                if col in x_dict:
                    value = x_dict[col]
                    if value in mapping_series.index:
                        result.append(mapping_series[value])
                    else:
                        # Handle unknown categories (use NaN mapping if available, otherwise -1)
                        if pd.isna(value) and np.nan in mapping_series.index:
                            result.append(mapping_series[np.nan])
                        else:
                            result.append(-1)
                else:
                    # Column not found
                    result.append(-1)

            return result

        return transform_one

    @register_step_handler(ce.HashingEncoder)
    def handle_hashing_encoder(step: Any) -> Callable[[Any], Any]:
        """Handle HashingEncoder for single-row input."""
        import hashlib

        def transform_one(x: Any) -> Any:
            if isinstance(x, (list, np.ndarray)):
                if hasattr(step, "cols") and step.cols:
                    x_dict = {col: x[i] for i, col in enumerate(step.cols)}
                else:
                    x_dict = dict(enumerate(x))
            else:
                x_dict = x  # type: ignore[unreachable]

            result = [0.0] * step.n_components

            # Hash each categorical column
            cols_to_process = (
                step.cols
                if hasattr(step, "cols") and step.cols
                else list(x_dict.keys())
            )

            for col in cols_to_process:
                if col in x_dict:
                    value = str(x_dict[col])
                    # Create hash (MD5 is used for non-security purposes - just data hashing)
                    hash_value = int(
                        hashlib.md5(value.encode(), usedforsecurity=False).hexdigest(),
                        16,
                    )  # nosec
                    idx = hash_value % step.n_components
                    result[idx] += 1.0

            return result

        return transform_one

    @register_step_handler(ce.TargetEncoder)
    def handle_target_encoder(step: Any) -> Callable[[Any], Any]:
        """Handle TargetEncoder for single-row input."""
        mapping = step.mapping

        def transform_one(x: Any) -> Any:
            if isinstance(x, (list, np.ndarray)):
                if hasattr(step, "cols") and step.cols:
                    x_dict = {col: x[i] for i, col in enumerate(step.cols)}
                else:
                    x_dict = dict(enumerate(x))
            else:
                x_dict = x  # type: ignore[unreachable]

            result = []

            # TargetEncoder mapping is different - it's a dict with column names as keys
            # and pandas Series as values
            for col in (
                step.cols if hasattr(step, "cols") and step.cols else x_dict.keys()
            ):
                if col in mapping and col in x_dict:
                    value = x_dict[col]
                    mapping_series = mapping[col]  # This is a pandas Series

                    # Find the ordinal value for this categorical value first
                    if hasattr(step, "ordinal_encoder") and step.ordinal_encoder:
                        # Get ordinal mapping for this column
                        ordinal_mapping = None
                        for ord_mapping in step.ordinal_encoder.mapping:
                            if ord_mapping["col"] == col:
                                ordinal_mapping = ord_mapping[
                                    "mapping"
                                ]  # pandas Series
                                break

                        if (
                            ordinal_mapping is not None
                            and value in ordinal_mapping.index
                        ):
                            ordinal_value = ordinal_mapping[value]
                            # Look up target encoding for this ordinal value
                            if ordinal_value in mapping_series.index:
                                result.append(mapping_series[ordinal_value])
                            else:
                                # Unknown ordinal value, use default
                                result.append(step._mean)
                        else:
                            # Unknown category, use overall mean
                            result.append(step._mean)
                    else:
                        # Fallback: try direct lookup in mapping
                        if value in mapping_series.index:
                            result.append(mapping_series[value])
                        else:
                            result.append(step._mean)
                elif col in x_dict:
                    # Column not in mapping, use mean
                    result.append(step._mean)

            return result

        return transform_one

    @register_step_handler(ce.CatBoostEncoder)
    def handle_catboost_encoder(step: Any) -> Callable[[Any], Any]:
        """Handle CatBoostEncoder for single-row input."""
        mapping = step.mapping

        def transform_one(x: Any) -> Any:
            if isinstance(x, (list, np.ndarray)):
                if hasattr(step, "cols") and step.cols:
                    x_dict = {col: x[i] for i, col in enumerate(step.cols)}
                else:
                    x_dict = dict(enumerate(x))
            else:
                x_dict = x  # type: ignore[unreachable]

            result = []

            # CatBoostEncoder mapping: dict with column names as keys, DataFrames as values
            # DataFrame has categorical values as index and sum/count columns
            # CatBoost uses smoothing: (sum + a * global_mean) / (count + a)
            a = getattr(step, "a", 1)  # smoothing parameter, default 1
            global_mean = step._mean

            for col in (
                step.cols if hasattr(step, "cols") and step.cols else x_dict.keys()
            ):
                if col in mapping and col in x_dict:
                    value = x_dict[col]
                    mapping_df = mapping[col]  # This is a DataFrame

                    if value in mapping_df.index:
                        # CatBoost encoding with smoothing: (sum + a * global_mean) / (count + a)
                        sum_val = mapping_df.loc[value, "sum"]
                        count_val = mapping_df.loc[value, "count"]
                        encoded_value = (sum_val + a * global_mean) / (count_val + a)
                        result.append(encoded_value)
                    else:
                        # Unknown category, use default
                        result.append(global_mean)
                elif col in x_dict:
                    # Column not in mapping, use default
                    result.append(global_mean)

            return result

        return transform_one

    @register_step_handler(ce.LeaveOneOutEncoder)
    def handle_leave_one_out_encoder(step: Any) -> Callable[[Any], Any]:
        """Handle LeaveOneOutEncoder for single-row input."""
        mapping = step.mapping

        def transform_one(x: Any) -> Any:
            if isinstance(x, (list, np.ndarray)):
                if hasattr(step, "cols") and step.cols:
                    x_dict = {col: x[i] for i, col in enumerate(step.cols)}
                else:
                    x_dict = dict(enumerate(x))
            else:
                x_dict = x  # type: ignore[unreachable]

            result = []

            # LeaveOneOutEncoder mapping: dict with column names as keys, DataFrames as values
            # DataFrame has categorical values as index and sum/count columns
            for col in (
                step.cols if hasattr(step, "cols") and step.cols else x_dict.keys()
            ):
                if col in mapping and col in x_dict:
                    value = x_dict[col]
                    mapping_df = mapping[col]  # This is a DataFrame

                    if value in mapping_df.index:
                        # Leave-one-out encoding: sum / count
                        sum_val = mapping_df.loc[value, "sum"]
                        count_val = mapping_df.loc[value, "count"]
                        encoded_value = (
                            sum_val / count_val if count_val > 0 else step._mean
                        )
                        result.append(encoded_value)
                    else:
                        # Unknown category, use overall mean
                        result.append(step._mean)
                elif col in x_dict:
                    # Column not in mapping, use mean
                    result.append(step._mean)

            return result

        return transform_one

    __all__ = [
        "handle_binary_encoder",
        "handle_ce_one_hot_encoder",
        "handle_ce_ordinal_encoder",
        "handle_hashing_encoder",
        "handle_target_encoder",
        "handle_catboost_encoder",
        "handle_leave_one_out_encoder",
        "_safe_fit_supervised_encoder",
    ]

else:
    # category_encoders not available
    __all__ = []
