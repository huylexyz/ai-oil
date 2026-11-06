"""Train TV4 Ridge and export Naive/Ridge Test forecasts for horizons 2–4.

Run ``python -m tv4.horizons_2_4`` from the repository root. The approved
1/5-horizon scripts, artifacts and prediction CSVs are never written here.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import PROCESSED_DATA_DIR
from src.features import FEATURE_COLUMNS
from tv4.ridge_regression import ALPHA_VALUES, FEATURES, make_pipeline, select_alpha

ROOT = Path(__file__).resolve().parent
HORIZONS = (2, 3, 4)
PREDICTION_COLUMNS = (
    "observation_date", "target_date", "horizon", "actual",
    "predicted", "model", "split",
)


def load_horizon_data(horizon: int) -> pd.DataFrame:
    """Read the official TV3 handoff with exactly the approved 29 inputs."""
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError(f"Horizon phải thuộc {HORIZONS}: {horizon!r}")
    if FEATURES != list(FEATURE_COLUMNS):
        raise ValueError("Thứ tự 29 đặc trưng TV4 và TV3 không khớp")
    path = PROCESSED_DATA_DIR / f"brent_features_h{horizon}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu dữ liệu TV3: {path}")
    date_col = f"target_date_{horizon}"
    target_col = f"target_{horizon}"
    split_col = f"split_{horizon}"
    expected = [
        "observation_date", *FEATURES, date_col, target_col,
        f"target_ret_{horizon}", split_col,
    ]
    data = pd.read_csv(path, parse_dates=["observation_date", date_col])
    if data.columns.tolist() != expected:
        raise ValueError(f"Sai tên hoặc thứ tự cột TV3 trong {path}")
    if data.empty or data.isna().any().any():
        raise ValueError(f"Dữ liệu TV3 rỗng hoặc thiếu giá trị: {path}")
    numeric = data[[*FEATURES, target_col, f"target_ret_{horizon}"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not data[target_col].gt(0).all():
        raise ValueError(f"Đặc trưng hoặc nhãn không hợp lệ: {path}")
    if data["observation_date"].duplicated().any() or not data[date_col].gt(data["observation_date"]).all():
        raise ValueError(f"Ngày quan sát/mục tiêu không hợp lệ: {path}")
    counts = data[split_col].value_counts().to_dict()
    if counts != {"train": 1986 - horizon, "val": 505, "test": 442}:
        raise ValueError(f"Số mẫu Train/Val/Test sai trong {path}: {counts}")
    for split, lower, upper in (
        ("train", None, pd.Timestamp("2022-12-31")),
        ("val", pd.Timestamp("2022-12-31"), pd.Timestamp("2024-12-31")),
        ("test", pd.Timestamp("2024-12-31"), None),
    ):
        dates = data.loc[data[split_col].eq(split), date_col]
        if (lower is not None and not dates.gt(lower).all()) or (upper is not None and not dates.le(upper).all()):
            raise ValueError(f"Ranh giới {split} sai theo ngày mục tiêu: {path}")
    return data


def make_test_predictions(test: pd.DataFrame, horizon: int, model: str, predicted) -> pd.DataFrame:
    """Use the seven-column TV4 prediction schema with official Test rows."""
    values = np.asarray(predicted, dtype=float)
    if len(values) != len(test) or not np.isfinite(values).all():
        raise ValueError(f"Dự báo {model} horizon {horizon} không hợp lệ")
    result = pd.DataFrame({
        "observation_date": test["observation_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "target_date": test[f"target_date_{horizon}"].dt.strftime("%Y-%m-%d").to_numpy(),
        "horizon": horizon,
        "actual": test[f"target_{horizon}"].to_numpy(dtype=float),
        "predicted": values,
        "model": model,
        "split": "test",
    })
    if len(result) != 442 or result.isna().any().any() or result["target_date"].duplicated().any():
        raise ValueError(f"CSV Test {model} horizon {horizon} không hợp lệ")
    return result.loc[:, list(PREDICTION_COLUMNS)]


def run_horizon(horizon: int):
    """Select alpha on Validation, fit the final model, and return outputs."""
    data = load_horizon_data(horizon)
    date_col = f"target_date_{horizon}"
    target_col = f"target_{horizon}"
    split_col = f"split_{horizon}"
    train = data.loc[data[split_col].eq("train")].copy()
    validation = data.loc[data[split_col].eq("val")].copy()
    test = data.loc[data[split_col].eq("test")].copy()

    first_validation_observation = validation["observation_date"].min()
    first_test_observation = test["observation_date"].min()
    removed_train = train.loc[train[date_col].gt(first_validation_observation)]
    removed_validation = validation.loc[validation[date_col].gt(first_test_observation)]
    if len(removed_train) != horizon - 1 or len(removed_validation) != horizon - 1:
        raise ValueError(f"Số mẫu purge horizon {horizon} không khớp ngày mục tiêu thực tế")
    train_tuning = train.loc[train[date_col].le(first_validation_observation)].copy()
    validation_final = validation.loc[validation[date_col].le(first_test_observation)].copy()
    if train_tuning.empty or validation_final.empty:
        raise ValueError(f"Không đủ mẫu sau purge horizon {horizon}")
    if train[date_col].max() >= validation[date_col].min() or validation[date_col].max() >= test[date_col].min():
        raise ValueError(f"Train/Validation/Test chồng lấn ngày mục tiêu horizon {horizon}")

    best_alpha, validation_mae, tuning_model = select_alpha(
        train_tuning, validation, target_col, horizon
    )
    if best_alpha not in ALPHA_VALUES:
        raise ValueError(f"Alpha horizon {horizon} không thuộc quy trình TV4")
    final_train = pd.concat([train, validation_final], ignore_index=True)
    if final_train[date_col].max() > first_test_observation:
        raise ValueError(f"Nhãn final fit horizon {horizon} vượt ranh giới Test")
    final_model = make_pipeline(best_alpha)
    final_model.fit(final_train[FEATURES], final_train[target_col])

    artifact = {
        "model": final_model,
        "horizon": horizon,
        "features": FEATURES,
        "alpha": best_alpha,
        "validation_mae": validation_mae,
        "training_split": "train + validation (labels available by first test observation)",
        "training_rows": len(final_train),
        "tuning_train_rows": len(train_tuning),
        "validation_rows_used_for_final_fit": len(validation_final),
        "excluded_train_boundary_rows_for_tuning": len(removed_train),
        "excluded_validation_boundary_rows_for_final_fit": len(removed_validation),
        "first_validation_observation_date": first_validation_observation.strftime("%Y-%m-%d"),
        "first_test_observation_date": first_test_observation.strftime("%Y-%m-%d"),
    }
    naive = make_test_predictions(test, horizon, "Naive", test["price"])
    ridge = make_test_predictions(test, horizon, "Ridge", final_model.predict(test[FEATURES]))
    return artifact, naive, ridge


def score_test(predictions: pd.DataFrame) -> tuple[float, float]:
    actual = predictions["actual"]
    predicted = predictions["predicted"]
    return float(mean_absolute_error(actual, predicted)), float(np.sqrt(mean_squared_error(actual, predicted)))


def main() -> None:
    destinations = {
        h: (
            ROOT / "models" / f"ridge_horizon_{h}.joblib",
            ROOT / "predictions" / f"naive_horizon_{h}.csv",
            ROOT / "predictions" / f"ridge_horizon_{h}.csv",
        )
        for h in HORIZONS
    }
    existing = [str(path) for paths in destinations.values() for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Không ghi đè sản phẩm TV4 đã có: {existing}")

    outputs = {h: run_horizon(h) for h in HORIZONS}
    for horizon, (artifact, naive, ridge) in outputs.items():
        model_path, naive_path, ridge_path = destinations[horizon]
        joblib.dump(artifact, model_path)
        naive.to_csv(naive_path, index=False)
        ridge.to_csv(ridge_path, index=False)
        for label, predictions in (("Naive", naive), ("Ridge", ridge)):
            mae, rmse = score_test(predictions)
            print(f"h={horizon} {label}: Test n={len(predictions)}, MAE={mae:.6f}, RMSE={rmse:.6f}")
        print(
            f"h={horizon} alpha={artifact['alpha']:g}, "
            f"purge Train/Val={artifact['excluded_train_boundary_rows_for_tuning']}, "
            f"Val/Test={artifact['excluded_validation_boundary_rows_for_final_fit']}"
        )


if __name__ == "__main__":
    main()
