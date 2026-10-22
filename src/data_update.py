"""Validated CSV updates for the maintained Brent observation set."""

import math
import os
import tempfile
from pathlib import Path

import pandas as pd

from .config import (
    CURRENT_DATA_PATH,
    DEFAULT_DATA_PATH,
    FORECAST_HORIZONS,
    MODEL_ARTIFACT_PATH,
    MODEL_DIR,
)
from .data_loader import DATE_COLUMN, PRICE_COLUMN, load_brent_data
from .features import PRICE_LAGS
from .train import train_forecasting_models


def _clean_upload(upload: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {DATE_COLUMN, PRICE_COLUMN}
    missing_columns = required.difference(upload.columns)
    if missing_columns:
        raise ValueError(
            "CSV thiếu cột bắt buộc: " + ", ".join(sorted(missing_columns))
        )

    frame = upload.loc[:, [DATE_COLUMN, PRICE_COLUMN]].copy()
    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        raise ValueError(f"CSV có {bad_dates} ngày trống hoặc sai định dạng.")
    frame[DATE_COLUMN] = dates
    frame[PRICE_COLUMN] = pd.to_numeric(frame[PRICE_COLUMN], errors="coerce")
    missing_prices = int(frame[PRICE_COLUMN].isna().sum())
    frame = frame.dropna(subset=[PRICE_COLUMN])
    if frame.empty:
        raise ValueError("CSV không có giá Brent dạng số hợp lệ.")
    if not frame[PRICE_COLUMN].map(lambda value: math.isfinite(value) and value > 0).all():
        raise ValueError("Giá Brent phải là số hữu hạn và lớn hơn 0.")

    conflicts = frame.groupby(DATE_COLUMN)[PRICE_COLUMN].nunique()
    if (conflicts > 1).any():
        raise ValueError("CSV có ngày trùng với các mức giá khác nhau.")
    duplicates_removed = int(frame[DATE_COLUMN].duplicated().sum())
    frame = (
        frame.drop_duplicates(DATE_COLUMN, keep="last")
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    return frame, {
        "missing_prices_skipped": missing_prices,
        "duplicate_rows_removed": duplicates_removed,
    }


def update_brent_data_and_models(upload: pd.DataFrame) -> dict:
    """Merge a validated CSV, train fresh inference models, and activate both.

    The original raw CSV and historical backtest outputs are never written by
    this function. Model training completes before the maintained CSV changes.
    """
    incoming, upload_quality = _clean_upload(upload)
    current_path = CURRENT_DATA_PATH if CURRENT_DATA_PATH.is_file() else DEFAULT_DATA_PATH
    current = load_brent_data(current_path)
    current_max = current[DATE_COLUMN].max()

    old_prices = current.set_index(DATE_COLUMN)[PRICE_COLUMN]
    incoming_overlap = incoming[incoming[DATE_COLUMN].isin(old_prices.index)].copy()
    revised = int(
        (incoming_overlap[PRICE_COLUMN].to_numpy()
         != incoming_overlap[DATE_COLUMN].map(old_prices).to_numpy()).sum()
    )
    added = incoming.loc[~incoming[DATE_COLUMN].isin(old_prices.index)].copy()
    newer = int((added[DATE_COLUMN] > current_max).sum())
    if added.empty and revised == 0:
        return {
            "updated": False,
            "message": "CSV không có quan sát mới hoặc giá lịch sử được điều chỉnh.",
            "latest_data_date": current_max.strftime("%Y-%m-%d"),
            **upload_quality,
        }

    merged = pd.concat([current, incoming], ignore_index=True)
    merged = (
        merged.drop_duplicates(DATE_COLUMN, keep="last")
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    if merged[DATE_COLUMN].duplicated().any():
        raise ValueError("Dữ liệu sau hợp nhất vẫn có ngày trùng.")
    minimum_rows = max(PRICE_LAGS) + max(FORECAST_HORIZONS) + 1
    if len(merged) < minimum_rows:
        raise ValueError(
            f"Cần tối thiểu {minimum_rows} quan sát hợp lệ để tạo đặc trưng và nhãn."
        )

    CURRENT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fd_data, temp_data_name = tempfile.mkstemp(
        prefix="brent_observations_", suffix=".csv", dir=CURRENT_DATA_PATH.parent
    )
    os.close(fd_data)
    temp_data = Path(temp_data_name)
    fd_model, temp_model_name = tempfile.mkstemp(
        prefix="forecast_models_", suffix=".joblib.tmp", dir=MODEL_DIR
    )
    os.close(fd_model)
    temp_model = Path(temp_model_name)
    previous_data = CURRENT_DATA_PATH.read_bytes() if CURRENT_DATA_PATH.is_file() else None

    try:
        merged.to_csv(temp_data, index=False, date_format="%Y-%m-%d")
        train_forecasting_models(merged, artifact_path=temp_model)

        # Activate the validated dataset, then its matching artifact. Roll back
        # the dataset if the artifact cannot be atomically activated.
        temp_data.replace(CURRENT_DATA_PATH)
        try:
            temp_model.replace(MODEL_ARTIFACT_PATH)
        except Exception:
            if previous_data is None:
                CURRENT_DATA_PATH.unlink(missing_ok=True)
            else:
                CURRENT_DATA_PATH.write_bytes(previous_data)
            raise
    finally:
        temp_data.unlink(missing_ok=True)
        temp_model.unlink(missing_ok=True)

    return {
        "updated": True,
        "rows_before": int(len(current)),
        "rows_after": int(len(merged)),
        "newer_observations": newer,
        "added_observations": int(len(added)),
        "revised_observations": revised,
        "latest_data_date": merged[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "artifact_path": str(MODEL_ARTIFACT_PATH),
        **upload_quality,
    }
