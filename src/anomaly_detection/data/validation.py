import pandas as pd

from ..utils.exceptions import DataValidationError


REQUIRED_COLS = {"timestamp", "value"}


def validate_series(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if df["timestamp"].isna().any():
        raise DataValidationError("Null timestamps found")
    if df["value"].isna().any():
        raise DataValidationError("Null values found")
    if not df["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamps not sorted")
    if df["value"].std() == 0:
        raise DataValidationError("Zero variance series")