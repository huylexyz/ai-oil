"""Load the handed-off Test forecasts into one Backtesting table."""

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
SOURCES = (
    ("Naive", 1, "tv4/predictions/naive_horizon_1.csv", "Naive"),
    ("Naive", 5, "tv4/predictions/naive_horizon_5.csv", "Naive"),
    ("Ridge", 1, "tv4/predictions/ridge_horizon_1.csv", "Ridge"),
    ("Ridge", 5, "tv4/predictions/ridge_horizon_5.csv", "Ridge"),
    ("Random Forest", 1, "tv5/results/rf_predictions_test_t1.csv", "RandomForest"),
    ("Random Forest", 5, "tv5/results/rf_predictions_test_t5.csv", "RandomForest"),
    ("XGBoost", 1, "tv6/results/xgb_predictions_test_h1.csv", "XGBoost"),
    ("XGBoost", 5, "tv6/results/xgb_predictions_test_h5.csv", "XGBoost"),
)


def _load_source(name: str, horizon: int, relative_path: str, raw_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu CSV dự báo: {path}")

    frame = pd.read_csv(path)
    missing = set(COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{relative_path}: thiếu cột {sorted(missing)}")

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
    frame["model"] = name
    frame["split"] = "test"

    if frame.isna().any().any():
        raise ValueError(f"{relative_path}: có giá trị thiếu trong dữ liệu Test")
    if not np.isfinite(frame[["actual", "predicted"]].to_numpy()).all():
        raise ValueError(f"{relative_path}: giá hoặc dự báo không hữu hạn")
    if not frame["target_date"].gt(frame["observation_date"]).all():
        raise ValueError(f"{relative_path}: ngày mục tiêu không sau ngày quan sát")
    return frame.reset_index(drop=True)


def load_backtest(split: str = "test") -> pd.DataFrame:
    """Return official forecasts for the requested split, with a fixed schema.

    Only Test is complete across all four models. TV4 has Validation rows,
    but TV5 and TV6 did not hand off Validation prediction CSVs.
    """
    if split == "validation":
        raise ValueError(
            "Chưa đủ dự báo Validation chính thức: thiếu Random Forest và "
            "XGBoost cho horizon 1 và 5. TV4 chỉ có Naive/Ridge (split='val')."
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

    for horizon in (1, 5):
        reference = frames[("Naive", horizon)].sort_values("target_date").reset_index(drop=True)
        for name in ("Ridge", "Random Forest", "XGBoost"):
            current = frames[(name, horizon)].sort_values("target_date").reset_index(drop=True)
            if not current[["observation_date", "target_date"]].equals(
                reference[["observation_date", "target_date"]]
            ):
                raise ValueError(f"{name} horizon {horizon}: ngày không khớp các mô hình")
            if not np.allclose(current["actual"], reference["actual"], rtol=0, atol=1e-9):
                raise ValueError(f"{name} horizon {horizon}: giá thực tế không khớp")

    return result.loc[:, list(COLUMNS)]
