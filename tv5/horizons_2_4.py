"""Train the approved TV5 Random Forest workflow for horizons 2, 3 and 4.

Run ``python -m tv5.horizons_2_4`` from the repository root. Existing
horizon 1 and 5 outputs are never written by this module.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.config import PROCESSED_DATA_DIR
from src.features import FEATURE_COLUMNS
from tv5.train_rf import PARAM_GRID, ROOT, make_features, rmse, summary_params

HORIZONS = (2, 3, 4)
PREDICTION_COLUMNS = (
    "observation_date", "target_date", "horizon", "actual",
    "predicted", "model", "split",
)


def load_horizon_data(horizon: int) -> pd.DataFrame:
    """Load the official TV3 file and require the exact 29 feature columns."""
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError(f"Horizon phải thuộc {HORIZONS}: {horizon!r}")
    path = PROCESSED_DATA_DIR / f"brent_features_h{horizon}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu dữ liệu TV3: {path}")
    target_date = f"target_date_{horizon}"
    target = f"target_{horizon}"
    split = f"split_{horizon}"
    expected = [
        "observation_date", *FEATURE_COLUMNS, target_date, target,
        f"target_ret_{horizon}", split,
    ]
    data = pd.read_csv(path, parse_dates=["observation_date", target_date])
    if data.columns.tolist() != expected:
        raise ValueError(f"Sai tên hoặc thứ tự cột TV3: {path}")
    if data.empty or data.isna().any().any():
        raise ValueError(f"Dữ liệu TV3 rỗng hoặc thiếu giá trị: {path}")
    if make_features(data).columns.tolist() != list(FEATURE_COLUMNS):
        raise ValueError(f"Đầu vào Random Forest không đúng 29 đặc trưng TV3: {path}")
    numbers = data[[*FEATURE_COLUMNS, target, f"target_ret_{horizon}"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or not data[target].gt(0).all():
        raise ValueError(f"Đặc trưng hoặc nhãn không hợp lệ: {path}")
    if (
        data["observation_date"].duplicated().any()
        or data[target_date].duplicated().any()
        or not data[target_date].gt(data["observation_date"]).all()
    ):
        raise ValueError(f"Ngày quan sát/mục tiêu không hợp lệ: {path}")
    counts = data[split].value_counts().to_dict()
    if counts != {"train": 1986 - horizon, "val": 505, "test": 442}:
        raise ValueError(f"Số mẫu Train/Val/Test sai: {path}: {counts}")
    for name, lower, upper in (
        ("train", None, pd.Timestamp("2022-12-31")),
        ("val", pd.Timestamp("2022-12-31"), pd.Timestamp("2024-12-31")),
        ("test", pd.Timestamp("2024-12-31"), None),
    ):
        dates = data.loc[data[split].eq(name), target_date]
        if (lower is not None and not dates.gt(lower).all()) or (
            upper is not None and not dates.le(upper).all()
        ):
            raise ValueError(f"Ranh giới {name} sai theo ngày mục tiêu: {path}")
    return data


def train_horizon(horizon: int):
    """Select on Validation, refit on available labels, then score Test."""
    data = load_horizon_data(horizon)
    target = f"target_{horizon}"
    target_date = f"target_date_{horizon}"
    split = f"split_{horizon}"
    train = data.loc[data[split].eq("train")].copy()
    val = data.loc[data[split].eq("val")].copy()
    test = data.loc[data[split].eq("test")].copy()
    first_val_observation = val["observation_date"].min()
    first_test_observation = test["observation_date"].min()
    train_purge = train[target_date].gt(first_val_observation)
    val_purge = val[target_date].gt(first_test_observation)
    if int(train_purge.sum()) != horizon - 1 or int(val_purge.sum()) != horizon - 1:
        raise ValueError(f"Số mẫu purge h{horizon} không khớp ngày mục tiêu thực tế")
    if train[target_date].max() >= val[target_date].min() or val[target_date].max() >= test[target_date].min():
        raise ValueError(f"Ranh giới Train/Validation/Test chồng lấn ở h{horizon}")
    train_fit = train.loc[~train_purge].copy()
    val_final = val.loc[~val_purge].copy()
    if train_fit.empty or val_final.empty:
        raise ValueError(f"Không đủ dữ liệu sau purge ở h{horizon}")
    if train_fit[target_date].max() > first_val_observation or val_final[target_date].max() > first_test_observation:
        raise ValueError(f"Nhãn vượt ranh giới thời gian ở h{horizon}")

    features = list(FEATURE_COLUMNS)
    X_train, y_train = make_features(train_fit), train_fit[target].astype(float)
    X_val, y_val = make_features(val), val[target].astype(float)
    if X_train.columns.tolist() != features or X_val.columns.tolist() != features:
        raise ValueError(f"Sai thứ tự 29 đặc trưng khi tuning h{horizon}")
    tuning_rows = []
    best = None
    start = time.perf_counter()
    for params in PARAM_GRID:
        candidate = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        candidate.fit(X_train, y_train)
        validation_prediction = candidate.predict(X_val)
        row = {
            "horizon": horizon,
            **params,
            "validation_MAE": float(mean_absolute_error(y_val, validation_prediction)),
            "validation_RMSE": rmse(y_val, validation_prediction),
        }
        tuning_rows.append(row)
        if best is None or row["validation_RMSE"] < best["validation_RMSE"]:
            best = row

    # Match the approved h5 protocol: all Train labels and only the Validation
    # labels available by the first Test observation enter the final fit.
    trainval = pd.concat([train, val_final]).sort_values("observation_date")
    if trainval[target_date].max() > first_test_observation:
        raise ValueError(f"Final fit dùng nhãn vượt ranh giới Test ở h{horizon}")
    X_trainval, y_trainval = make_features(trainval), trainval[target].astype(float)
    X_test = make_features(test)
    if X_trainval.columns.tolist() != features or X_test.columns.tolist() != features:
        raise ValueError(f"Sai thứ tự 29 đặc trưng khi final fit h{horizon}")
    params = summary_params(best)
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_trainval, y_trainval)
    predicted = model.predict(X_test)
    if model.n_features_in_ != 29 or model.feature_names_in_.tolist() != features:
        raise ValueError(f"Artifact h{horizon} có sai đặc trưng")

    predictions = pd.DataFrame({
        "observation_date": test["observation_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "target_date": test[target_date].dt.strftime("%Y-%m-%d").to_numpy(),
        "horizon": horizon,
        "actual": test[target].to_numpy(dtype=float),
        "predicted": predicted,
        "model": "RandomForest",
        "split": "test",
    }).loc[:, list(PREDICTION_COLUMNS)]
    if len(predictions) != 442 or predictions.isna().any().any() or predictions["target_date"].duplicated().any():
        raise ValueError(f"CSV Test h{horizon} không hợp lệ")
    test_mae = float(mean_absolute_error(predictions["actual"], predictions["predicted"]))
    test_rmse = rmse(predictions["actual"], predictions["predicted"])
    purge = {
        "enabled": True,
        "train_val": int(train_purge.sum()),
        "val_test": int(val_purge.sum()),
        "train_val_observation_dates": train.loc[train_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
        "val_test_observation_dates": val.loc[val_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
    }
    model_file = f"models/rf_model_t{horizon}.joblib"
    metadata_file = f"models/rf_model_t{horizon}_metadata.json"
    prediction_file = f"results/rf_predictions_test_t{horizon}.csv"
    common = {
        "horizon": horizon,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "purged_boundary_samples": purge,
        "train_rows_used_for_tuning": len(train_fit),
        "trainval_rows_used_for_final_fit": len(trainval),
        "best_params": params,
        "best_validation_MAE": best["validation_MAE"],
        "best_validation_RMSE": best["validation_RMSE"],
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
        "features": features,
        "test_predictions_file": prediction_file,
        "model_file": model_file,
    }
    summary = {
        **common,
        "fit_seconds_including_tuning": round(time.perf_counter() - start, 2),
        "metadata_file": metadata_file,
    }
    metadata = {
        "model": "RandomForestRegressor",
        "target_column": target,
        "n_features": len(features),
        "random_state": 42,
        "selection_metric": "validation_RMSE",
        "fit_data": "train + validation (da purge ranh gioi Val/Test)",
        **common,
        "trainval_observation_range": [
            trainval["observation_date"].min().strftime("%Y-%m-%d"),
            trainval["observation_date"].max().strftime("%Y-%m-%d"),
        ],
        "trainval_target_date_range": [
            trainval[target_date].min().strftime("%Y-%m-%d"),
            trainval[target_date].max().strftime("%Y-%m-%d"),
        ],
        "test_observation_range": [
            test["observation_date"].min().strftime("%Y-%m-%d"),
            test["observation_date"].max().strftime("%Y-%m-%d"),
        ],
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return model, metadata, summary, pd.DataFrame(tuning_rows), predictions


def main() -> None:
    destinations = {
        h: (
            ROOT / "models" / f"rf_model_t{h}.joblib",
            ROOT / "models" / f"rf_model_t{h}_metadata.json",
            ROOT / "results" / f"rf_predictions_test_t{h}.csv",
            ROOT / "results" / f"rf_tuning_h{h}.csv",
            ROOT / "results" / f"rf_summary_t{h}.json",
        )
        for h in HORIZONS
    }
    existing = [str(path) for paths in destinations.values() for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Không ghi đè sản phẩm TV5 đã có: {existing}")
    outputs = {h: train_horizon(h) for h in HORIZONS}
    for h, (model, metadata, summary, tuning, predictions) in outputs.items():
        model_path, metadata_path, prediction_path, tuning_path, summary_path = destinations[h]
        joblib.dump(model, model_path)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        predictions.to_csv(prediction_path, index=False)
        tuning.to_csv(tuning_path, index=False)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"h{h}: train/val/test={summary['train_rows']}/{summary['validation_rows']}/{summary['test_rows']}, "
            f"purge={h-1}/{h-1}, tuning_train={summary['train_rows_used_for_tuning']}, "
            f"final_fit={summary['trainval_rows_used_for_final_fit']}, params={summary['best_params']}, "
            f"Test MAE={summary['test_MAE']:.6f}, RMSE={summary['test_RMSE']:.6f}"
        )


if __name__ == "__main__":
    main()
