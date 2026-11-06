"""Load the handed-off horizon 1–5 Test forecasts into one Backtesting table."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

COLUMNS = (
    "observation_date",
    "target_date",
    "horizon",
    "actual",
    "predicted",
    "model",
    "split",
)
EXPECTED_TEST_ROWS = 442
HORIZONS = (1, 2, 3, 4, 5)
SOURCES = tuple(
    (name, horizon, pattern.format(h=horizon), raw_name)
    for name, pattern, raw_name in (
        ("Naive", "tv4/predictions/naive_horizon_{h}.csv", "Naive"),
        ("Ridge", "tv4/predictions/ridge_horizon_{h}.csv", "Ridge"),
        ("Random Forest", "tv5/results/rf_predictions_test_t{h}.csv", "RandomForest"),
        ("XGBoost", "tv6/results/xgb_predictions_test_h{h}.csv", "XGBoost"),
    )
    for horizon in HORIZONS
)


def _load_source(name: str, horizon: int, relative_path: str, raw_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu CSV dự báo: {path}")

    frame = pd.read_csv(path)
    if frame.columns.tolist() != list(COLUMNS):
        missing = set(COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{relative_path}: thiếu cột {sorted(missing)}")
        raise ValueError(f"{relative_path}: sai schema hoặc thứ tự cột CSV")

    frame = frame.loc[frame["split"].eq("test"), list(COLUMNS)].copy()
    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"{relative_path}: cần {EXPECTED_TEST_ROWS} dòng Test, thấy {len(frame)}"
        )
    if not frame["model"].eq(raw_name).all():
        raise ValueError(f"{relative_path}: tên mô hình không khớp {raw_name}")

    for column in ("observation_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in ("horizon", "actual", "predicted"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not frame["horizon"].eq(horizon).all():
        raise ValueError(f"{relative_path}: horizon không khớp {horizon}")
    frame["horizon"] = frame["horizon"].astype("int64")
    frame[["actual", "predicted"]] = frame[["actual", "predicted"]].astype("float64")
    frame["model"] = name
    frame["split"] = "test"

    if frame.isna().any().any():
        raise ValueError(f"{relative_path}: có giá trị thiếu trong dữ liệu Test")
    if not np.isfinite(frame[["actual", "predicted"]].to_numpy()).all():
        raise ValueError(f"{relative_path}: giá hoặc dự báo không hữu hạn")
    if frame[["actual", "predicted"]].le(0).any().any():
        raise ValueError(f"{relative_path}: giá hoặc dự báo phải dương")
    if not frame["target_date"].gt(frame["observation_date"]).all():
        raise ValueError(f"{relative_path}: ngày mục tiêu không sau ngày quan sát")
    return frame.reset_index(drop=True)


def load_backtest(split: str = "test") -> pd.DataFrame:
    """Return official forecasts for the requested split, with a fixed schema.

    Only Test is complete across all four models and five horizons.
    """
    if split == "validation":
        raise ValueError(
            "Chưa đủ dự báo Validation chính thức cho bốn mô hình và horizon 1–5: "
            "thiếu Random Forest và XGBoost cho horizon 1 và 5; "
            "các CSV Backtesting horizon 2–4 chỉ đầy đủ trên Test. "
            "TV4 chỉ có Naive/Ridge Validation ở horizon 1 và 5 (split='val')."
        )
    if split != "test":
        raise ValueError("split chỉ hỗ trợ 'test'; 'validation' hiện chưa đủ dữ liệu")

    frames = {
        (name, horizon): _load_source(name, horizon, path, raw_name)
        for name, horizon, path, raw_name in SOURCES
    }
    result = pd.concat(frames.values(), ignore_index=True)
    key = ["model", "horizon", "split", "target_date"]
    if result.duplicated(key).any():
        raise ValueError(f"Dự báo trùng khóa {key}")

    reference_target_dates = None
    reference_actual = None
    for horizon in HORIZONS:
        reference = frames[("Naive", horizon)].sort_values("target_date").reset_index(drop=True)
        for name in ("Ridge", "Random Forest", "XGBoost"):
            current = frames[(name, horizon)].sort_values("target_date").reset_index(drop=True)
            if not current[["observation_date", "target_date"]].equals(
                reference[["observation_date", "target_date"]]
            ):
                raise ValueError(f"{name} horizon {horizon}: ngày không khớp các mô hình")
            if not np.allclose(current["actual"], reference["actual"], rtol=0, atol=1e-9):
                raise ValueError(f"{name} horizon {horizon}: giá thực tế không khớp")
        if reference_target_dates is None:
            reference_target_dates = reference["target_date"]
            reference_actual = reference["actual"]
        else:
            if not reference["target_date"].equals(reference_target_dates):
                raise ValueError(f"Horizon {horizon}: ngày mục tiêu Test không khớp horizon 1")
            if not np.allclose(reference["actual"], reference_actual, rtol=0, atol=1e-9):
                raise ValueError(f"Horizon {horizon}: giá thực tế Test không khớp horizon 1")

    return result.loc[:, list(COLUMNS)]
