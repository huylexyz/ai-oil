"""Forecast metrics for the project handoff schema."""

import numpy as np
import pandas as pd

from .config import RESULT_COLUMNS


def evaluate(forecasts: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Calculate MAE, RMSE and direction accuracy by model/horizon/split."""
    missing = set(RESULT_COLUMNS).difference(forecasts.columns)
    if missing:
        raise ValueError(f"Thiếu cột kết quả: {', '.join(sorted(missing))}")

    origin_prices = features.set_index("origin_date")["actual"]
    scored = forecasts.copy()
    scored["origin_actual"] = scored["origin_date"].map(origin_prices)
    if scored["origin_actual"].isna().any():
        raise ValueError("Không tìm thấy giá tại một hoặc nhiều ngày dự báo.")

    rows = []
    for (model, horizon, split), group in forecasts.groupby(
        ["model", "horizon", "split"], sort=True
    ):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        # Direction is measured from the known price at the forecast origin.
        origin = scored.loc[group.index, "origin_actual"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "split": split,
                "n": len(group),
                "MAE": float(np.mean(np.abs(actual - predicted))),
                "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                # Persistence forecasts no movement, so it is not a directional
                # forecast and must not be assigned a misleading near-zero DA.
                "Directional Accuracy": (
                    float(np.mean(np.sign(actual - origin) == np.sign(predicted - origin)))
                    if model != "Naive Persistence"
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
