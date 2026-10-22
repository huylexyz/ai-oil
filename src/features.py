"""Leakage-safe lagged features and forward targets for Brent prices."""

import pandas as pd

from .config import FORECAST_HORIZONS
from .data_loader import DATE_COLUMN, PRICE_COLUMN

# Values at the forecast origin are known when the forecast is made. Rolling
# windows are backward-looking and include the origin date only.
PRICE_LAGS = (0, 1, 2, 3, 5, 10, 20)
RETURN_LAGS = (1, 2, 5, 10, 20)
ROLLING_WINDOWS = (5, 10, 20)


def feature_columns() -> list[str]:
    """Return the ordered, agreed feature schema used by all models."""
    return (
        [f"price_lag_{lag}" for lag in PRICE_LAGS]
        + [f"return_lag_{lag}" for lag in RETURN_LAGS]
        + [f"price_mean_{window}" for window in ROLLING_WINDOWS]
        + [f"price_std_{window}" for window in ROLLING_WINDOWS]
    )


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build origin-date features and targets one/five valid observations ahead.

    The input must contain valid, sorted observations. Targets are shifted over
    valid observations, so a horizon is not a calendar-day interval.
    """
    frame = data.loc[:, [DATE_COLUMN, PRICE_COLUMN]].copy()
    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    price = frame[PRICE_COLUMN].astype(float)
    features = pd.DataFrame({DATE_COLUMN: frame[DATE_COLUMN]})

    for lag in PRICE_LAGS:
        features[f"price_lag_{lag}"] = price.shift(lag)
    returns = price.pct_change(fill_method=None)
    for lag in RETURN_LAGS:
        features[f"return_lag_{lag}"] = returns.shift(lag - 1)
    for window in ROLLING_WINDOWS:
        features[f"price_mean_{window}"] = price.rolling(window, min_periods=window).mean()
        features[f"price_std_{window}"] = price.rolling(window, min_periods=window).std()

    features["actual"] = price
    features["origin_date"] = frame[DATE_COLUMN]
    for horizon in FORECAST_HORIZONS:
        features[f"target_{horizon}"] = price.shift(-horizon)
        features[f"target_date_{horizon}"] = frame[DATE_COLUMN].shift(-horizon)

    # Keep the final rows here: each horizon has its own available target tail.
    # The forecaster removes missing labels separately for each horizon.
    return features.dropna(subset=feature_columns()).reset_index(drop=True)
