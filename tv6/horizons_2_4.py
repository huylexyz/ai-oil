"""Tune and finalize the approved TV6 XGBoost workflow for h2, h3 and h4.

Run ``python -m tv6.horizons_2_4`` from the repository root. This module
never writes the accepted horizon 1/5 products or their combined reports.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from src.config import PROCESSED_DATA_DIR
from src.features import FEATURE_COLUMNS
from tv6.tune_xgb import scores

ROOT = Path(__file__).resolve().parent
HORIZONS = (2, 3, 4)
PREDICTION_COLUMNS = (
    "observation_date", "target_date", "horizon", "actual",
    "predicted", "model", "split",
)
CANDIDATES = tuple(product((2, 3, 5), (0.03, 0.05), (3, 5, 10)))


def _json_safe(value):
    """Represent XGBoost's unset ``missing=NaN`` as JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def load_horizon_data(horizon: int) -> tuple[Path, pd.DataFrame]:
    """Read one TV3 handoff with exactly the approved 29 model features."""
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
    numbers = data[[*FEATURE_COLUMNS, target, f"target_ret_{horizon}"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or not data[target].gt(0).all():
        raise ValueError(f"Đặc trưng hoặc nhãn không hợp lệ: {path}")
    if (
        data["observation_date"].duplicated().any()
        or data[target_date].duplicated().any()
        or not data[target_date].gt(data["observation_date"]).all()
        or not data["observation_date"].is_monotonic_increasing
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
    return path, data


def _candidate(depth: int, rate: float, child_weight: int) -> XGBRegressor:
    """Use the exact fixed settings of TV6's 18-trial Validation tuner."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        max_depth=depth,
        learning_rate=rate,
        min_child_weight=child_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
        importance_type="gain",
        random_state=42,
        n_jobs=4,
    )


def run_horizon(horizon: int) -> dict:
    """Select on Validation and fit only labels known by the first Test row."""
    data_path, data = load_horizon_data(horizon)
    features = list(FEATURE_COLUMNS)
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
    train_tuning = train.loc[~train_purge].copy()
    val_final = val.loc[~val_purge].copy()
    if train_tuning.empty or val_final.empty:
        raise ValueError(f"Không đủ dữ liệu sau purge ở h{horizon}")
    if train_tuning[target_date].max() > first_val_observation or val_final[target_date].max() > first_test_observation:
        raise ValueError(f"Nhãn vượt ranh giới thời gian ở h{horizon}")

    X_train, y_train = train_tuning[features], train_tuning[target]
    X_val, y_val = val[features], val[target]
    rows = []
    best_row = None
    best_model = None
    for trial, (depth, rate, child_weight) in enumerate(CANDIDATES, start=1):
        model = _candidate(depth, rate, child_weight)
        start = time.perf_counter()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        result = scores(y_val, model.predict(X_val))
        row = {
            "horizon": horizon,
            "trial": trial,
            "max_depth": depth,
            "learning_rate": rate,
            "min_child_weight": child_weight,
            "best_n_estimators": int(model.best_iteration + 1),
            "validation_MAE": result["MAE"],
            "validation_RMSE": result["RMSE"],
            "fit_seconds": round(time.perf_counter() - start, 3),
        }
        rows.append(row)
        if best_row is None or row["validation_RMSE"] < best_row["validation_RMSE"]:
            best_row = row.copy()
            best_model = model

    tuning_table = pd.DataFrame(rows).sort_values(["validation_RMSE", "trial"])
    selected_validation = pd.DataFrame({
        "observation_date": val["observation_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "target_date": val[target_date].dt.strftime("%Y-%m-%d").to_numpy(),
        "horizon": horizon,
        "actual": y_val.to_numpy(dtype=float),
        "predicted": best_model.predict(X_val),
        "model": "XGBoost",
        "split": "val",
    }).loc[:, list(PREDICTION_COLUMNS)]
    selected_artifact = {
        "model": best_model,
        "features": features,
        "horizon": horizon,
        "stage": "validation_tuning",
    }
    tuning_summary = {
        "stage": "validation_tuning",
        "horizon": horizon,
        "train_rows_before_purge": len(train),
        "train_rows": len(train_tuning),
        "validation_rows": len(val),
        "purged_train_boundary_rows": int(train_purge.sum()),
        "purged_train_observation_dates": train.loc[train_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
        "first_validation_observation_date": first_val_observation.strftime("%Y-%m-%d"),
        "selection_metric": "validation_RMSE",
        "best_trial": best_row,
        "selected_model_params": best_model.get_params(),
        "naive_validation": scores(y_val, val["price"]),
        "features": features,
    }

    # The official finalizer drops early stopping and uses the selected
    # best_iteration + 1 as a fixed tree count on Train + safe Validation.
    trainval = pd.concat([train, val_final]).sort_values("observation_date")
    if trainval[target_date].max() > first_test_observation:
        raise ValueError(f"Final fit dùng nhãn vượt ranh giới Test ở h{horizon}")
    params = best_model.get_params().copy()
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = int(best_row["best_n_estimators"])
    final_model = XGBRegressor(**params)
    start = time.perf_counter()
    final_model.fit(trainval[features], trainval[target], verbose=False)
    final_fit_seconds = round(time.perf_counter() - start, 3)
    predicted = final_model.predict(test[features])
    if not np.isfinite(predicted).all() or final_model.feature_names_in_.tolist() != features:
        raise ValueError(f"Model hoặc dự báo h{horizon} không hợp lệ")
    final_artifact = {
        "model": final_model,
        "features": features,
        "horizon": horizon,
        "stage": "final_trainval",
        "params": final_model.get_params(),
        "first_test_origin": first_test_observation.strftime("%Y-%m-%d"),
        "training_target_date_max": trainval[target_date].max().strftime("%Y-%m-%d"),
        "xgboost_version": xgboost.__version__,
    }
    predictions = pd.DataFrame({
        "observation_date": test["observation_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "target_date": test[target_date].dt.strftime("%Y-%m-%d").to_numpy(),
        "horizon": horizon,
        "actual": test[target].to_numpy(dtype=float),
        "predicted": predicted,
        "model": "XGBoost",
        "split": "test",
    }).loc[:, list(PREDICTION_COLUMNS)]
    if len(predictions) != 442 or predictions.isna().any().any() or predictions["target_date"].duplicated().any():
        raise ValueError(f"CSV Test h{horizon} không hợp lệ")
    test_scores = scores(predictions["actual"], predictions["predicted"])
    final_summary = {
        "horizon": horizon,
        "stage": "final_trainval",
        "selection_metric": "validation_RMSE",
        "selected_validation_MAE": best_row["validation_MAE"],
        "selected_validation_RMSE": best_row["validation_RMSE"],
        "train_rows_before_purge": len(train),
        "train_rows_used_for_tuning": len(train_tuning),
        "validation_rows_before_purge": len(val),
        "validation_rows_used_for_final_fit": len(val_final),
        "purged_train_boundary_rows": int(train_purge.sum()),
        "purged_validation_boundary_rows": int(val_purge.sum()),
        "purged_train_observation_dates": tuning_summary["purged_train_observation_dates"],
        "purged_validation_observation_dates": val.loc[val_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
        "first_validation_observation_date": first_val_observation.strftime("%Y-%m-%d"),
        "trainval_rows_before_filter": len(train) + len(val),
        "removed_boundary_rows": int(val_purge.sum()),
        "final_training_rows": len(trainval),
        "test_rows": len(test),
        "first_test_origin": first_test_observation.strftime("%Y-%m-%d"),
        "training_target_date_max": trainval[target_date].max().strftime("%Y-%m-%d"),
        "n_estimators": params["n_estimators"],
        "fit_seconds": final_fit_seconds,
        "params": final_model.get_params(),
        "features": features,
        "model_file": f"models/xgb_horizon_{horizon}.joblib",
        "predictions_file": f"results/xgb_predictions_test_h{horizon}.csv",
        "test_MAE": test_scores["MAE"],
        "test_RMSE": test_scores["RMSE"],
    }
    report = {
        "data_file": str(data_path.relative_to(ROOT.parent)).replace("\\", "/"),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "horizons": [final_summary],
    }
    return {
        "tuning_table": tuning_table,
        "selected_artifact": selected_artifact,
        "selected_validation": selected_validation,
        "tuning_summary": tuning_summary,
        "final_artifact": final_artifact,
        "predictions": predictions,
        "report": report,
    }


def main() -> None:
    destinations = {
        h: (
            ROOT / "results" / "tuning" / f"xgb_tuning_h{h}.csv",
            ROOT / "results" / "tuning" / f"xgb_selected_h{h}.joblib",
            ROOT / "results" / "tuning" / f"xgb_selected_validation_h{h}.csv",
            ROOT / "results" / "tuning" / f"xgb_tuning_summary_h{h}.json",
            ROOT / "models" / f"xgb_horizon_{h}.joblib",
            ROOT / "results" / f"xgb_predictions_test_h{h}.csv",
            ROOT / "results" / f"xgb_summary_h{h}.json",
        )
        for h in HORIZONS
    }
    existing = [str(path) for paths in destinations.values() for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Không ghi đè sản phẩm TV6 đã có: {existing}")
    outputs = {h: run_horizon(h) for h in HORIZONS}
    for h, output in outputs.items():
        paths = destinations[h]
        output["tuning_table"].to_csv(paths[0], index=False)
        joblib.dump(output["selected_artifact"], paths[1])
        output["selected_validation"].to_csv(paths[2], index=False)
        paths[3].write_text(
            json.dumps(_json_safe(output["tuning_summary"]), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        joblib.dump(output["final_artifact"], paths[4])
        loaded = joblib.load(paths[4])
        _, data = load_horizon_data(h)
        test = data.loc[data[f"split_{h}"].eq("test")]
        recreated = loaded["model"].predict(test[loaded["features"]])
        np.testing.assert_allclose(output["predictions"]["predicted"], recreated, rtol=1e-7, atol=1e-7)
        output["predictions"].to_csv(paths[5], index=False)
        paths[6].write_text(
            json.dumps(_json_safe(output["report"]), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        summary = output["report"]["horizons"][0]
        print(
            f"h{h}: train={summary['train_rows_before_purge']}->{summary['train_rows_used_for_tuning']}, "
            f"val={summary['validation_rows_before_purge']}->{summary['validation_rows_used_for_final_fit']}, "
            f"best=(depth={summary['params']['max_depth']}, lr={summary['params']['learning_rate']}, "
            f"child={summary['params']['min_child_weight']}, trees={summary['n_estimators']}), "
            f"Validation RMSE={summary['selected_validation_RMSE']:.6f}, "
            f"Test MAE={summary['test_MAE']:.6f}, RMSE={summary['test_RMSE']:.6f}"
        )


if __name__ == "__main__":
    main()
