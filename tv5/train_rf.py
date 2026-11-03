"""
TV5 - Random Forest cho du bao gia dau Brent 1 va 5 phien.
Input: brent_features.csv do TV3 ban giao.
Quy uoc:
- Tune tham so chi tren Validation.
- Chon feature chi tu thong tin tai observation_date; loai tat ca target_* va split_* khoi X.
- Horizon 5: purge 4 mau ranh gioi Train/Validation khi tuning va 4 mau ranh gioi
  Validation/Test khi fit cuoi (nhan 5 phien chong lan giua cac tap).
- Sau khi chon tham so, fit lai tren Train + Validation.
- Xuat predictions tren Test (co cot split) va metadata mo hinh de TV8 danh gia doc lap.
"""
from pathlib import Path
import json
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
import joblib
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "data" / "processed" / "brent_features.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Tham so thu tren Validation. Co the mo rong sau khi chay ban dau.
PARAM_GRID = [
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
    {"n_estimators": 200, "max_depth": 12,   "min_samples_leaf": 1, "max_features": 1.0},
    {"n_estimators": 200, "max_depth": 24,   "min_samples_leaf": 1, "max_features": 1.0},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "max_features": 1.0},
    {"n_estimators": 200, "max_depth": 12,   "min_samples_leaf": 2, "max_features": 1.0},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.8},
]

DATE_COL = "observation_date"
# Exclude all labels, future dates/returns, and split metadata from model inputs.
EXCLUDE_PREFIXES = ("target_", "split_")
EXCLUDE_EXACT = {DATE_COL, "target_date_1", "target_date_5"}

# Horizon co nhan chong lan (target nhin truoc h phien) can purge o ranh gioi giua cac tap.
# Quy tac: mau o tap truoc bi loai neu target_date cua no > observation_date dau tien cua tap sau.
# Voi h=5 quy tac nay loai h-1 = 4 mau moi ranh gioi.
PURGE_HORIZONS = {5}
SPLIT_NAME = "test"

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_features(df):
    excluded = set(EXCLUDE_EXACT)
    excluded.update(c for c in df.columns if c.startswith(EXCLUDE_PREFIXES))
    X = df[[c for c in df.columns if c not in excluded]].copy()
    # Model only accepts numeric columns. Fail loudly if TV3 adds an unexpected text field.
    bad = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if bad:
        raise ValueError(f"Feature khong phai so can xu ly: {bad}")
    return X

def summary_params(best):
    return {
        k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
        for k, v in best.items()
        if k not in ("horizon", "validation_MAE", "validation_RMSE")
    }

def preds_path_name(horizon):
    return Path("results") / f"rf_predictions_test_t{horizon}.csv"

def tune_and_predict(df, horizon):
    target_col = f"target_{horizon}"
    split_col = f"split_{horizon}"
    target_date_col = f"target_date_{horizon}"

    work = df.dropna(subset=[target_col, split_col, target_date_col]).copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    work[target_date_col] = pd.to_datetime(work[target_date_col])

    train = work[work[split_col] == "train"].copy()
    val = work[work[split_col] == "val"].copy()
    test = work[work[split_col] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"Train/val/test rong cho horizon={horizon}")

    # Guard against target-date boundary leakage.
    if train[target_date_col].max() >= val[target_date_col].min():
        raise ValueError(f"Ranh gioi Train/Validation khong hop le cho horizon={horizon}")
    if val[target_date_col].max() >= test[target_date_col].min():
        raise ValueError(f"Ranh gioi Validation/Test khong hop le cho horizon={horizon}")

    # Purge ranh gioi Train/Validation (tuning) va Validation/Test (fit cuoi).
    purged_info = {"enabled": horizon in PURGE_HORIZONS, "train_val": 0, "val_test": 0}
    train_fit, val_final = train, val
    if horizon in PURGE_HORIZONS:
        tv_mask = train[target_date_col] > val[DATE_COL].min()
        vt_mask = val[target_date_col] > test[DATE_COL].min()
        purged_info["train_val"] = int(tv_mask.sum())
        purged_info["val_test"] = int(vt_mask.sum())
        purged_info["train_val_observation_dates"] = train.loc[tv_mask, DATE_COL].dt.strftime("%Y-%m-%d").tolist()
        purged_info["val_test_observation_dates"] = val.loc[vt_mask, DATE_COL].dt.strftime("%Y-%m-%d").tolist()
        expected = horizon - 1
        if purged_info["train_val"] != expected or purged_info["val_test"] != expected:
            raise ValueError(
                f"So mau purge bat thuong cho horizon={horizon}: {purged_info} (ky vong {expected} moi ranh gioi)"
            )
        train_fit = train[~tv_mask].copy()
        val_final = val[~vt_mask].copy()

    X_train, y_train = make_features(train_fit), train_fit[target_col].astype(float)
    X_val, y_val = make_features(val), val[target_col].astype(float)
    if X_train.columns.tolist() != X_val.columns.tolist():
        raise ValueError("Danh sach feature Train va Validation khong dong nhat")

    best = None
    tuning_rows = []
    start = time.time()
    for params in PARAM_GRID:
        model = RandomForestRegressor(
            **params, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        row = {
            "horizon": horizon,
            **params,
            "validation_MAE": mean_absolute_error(y_val, pred),
            "validation_RMSE": rmse(y_val, pred),
        }
        tuning_rows.append(row)
        # Select the configuration with the lowest validation RMSE.
        if best is None or row["validation_RMSE"] < best["validation_RMSE"]:
            best = row

    pd.DataFrame(tuning_rows).to_csv(
        RESULT_DIR / f"rf_tuning_h{horizon}.csv", index=False
    )

    # Refit selected model on Train + Validation (da purge ranh gioi Val/Test), then predict untouched Test.
    trainval = pd.concat([train, val_final]).sort_values(DATE_COL)
    X_trainval, y_trainval = make_features(trainval), trainval[target_col].astype(float)
    X_test = make_features(test)
    if X_trainval.columns.tolist() != X_test.columns.tolist():
        raise ValueError("Danh sach feature Train+Val va Test khong dong nhat")
    final_model = RandomForestRegressor(
        n_estimators=int(best["n_estimators"]),
        max_depth=None if pd.isna(best["max_depth"]) else int(best["max_depth"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        max_features=best["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_trainval, y_trainval)
    y_pred = final_model.predict(X_test)
    fit_seconds = round(time.time() - start, 2)

    model_path = MODEL_DIR / f"rf_model_t{horizon}.joblib"
    metadata_path = MODEL_DIR / f"rf_model_t{horizon}_metadata.json"
    dump(final_model, model_path)

    preds = pd.DataFrame({
        "observation_date": test[DATE_COL].dt.strftime("%Y-%m-%d").values,
        "target_date": test[target_date_col].dt.strftime("%Y-%m-%d").values,
        "horizon": horizon,
        "actual": test[target_col].astype(float).values,
        "predicted": y_pred,
        "model": "RandomForest",
        "split": SPLIT_NAME,
    })
    preds.to_csv(RESULT_DIR / f"rf_predictions_test_t{horizon}.csv", index=False)

    # Save validation metrics from the tuning stage (not test metrics).
    summary = {
        "horizon": horizon,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "purged_boundary_samples": purged_info,
        "train_rows_used_for_tuning": int(len(train_fit)),
        "trainval_rows_used_for_final_fit": int(len(trainval)),
        "best_params": summary_params(best),
        "best_validation_MAE": float(best["validation_MAE"]),
        "best_validation_RMSE": float(best["validation_RMSE"]),
        "fit_seconds_including_tuning": fit_seconds,
        "features": X_train.columns.tolist(),
        "test_predictions_file": str(preds_path_name(horizon)),
        "model_file": str(model_path.relative_to(ROOT)),
        "metadata_file": str(metadata_path.relative_to(ROOT)),
    }

    # Metadata di kem model: du de nap lai model va kiem tra dung dac trung/thu tu cot.
    metadata = {
        "model": "RandomForestRegressor",
        "horizon": horizon,
        "target_column": target_col,
        "features": X_train.columns.tolist(),
        "n_features": int(X_train.shape[1]),
        "best_params": summary_params(best),
        "random_state": 42,
        "selection_metric": "validation_RMSE",
        "best_validation_MAE": float(best["validation_MAE"]),
        "best_validation_RMSE": float(best["validation_RMSE"]),
        "fit_data": "train + validation (da purge ranh gioi Val/Test)",
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_rows_used_for_tuning": int(len(train_fit)),
        "trainval_rows_used_for_final_fit": int(len(trainval)),
        "purged_boundary_samples": purged_info,
        "trainval_observation_range": [
            trainval[DATE_COL].min().strftime("%Y-%m-%d"), trainval[DATE_COL].max().strftime("%Y-%m-%d")],
        "trainval_target_date_range": [
            trainval[target_date_col].min().strftime("%Y-%m-%d"), trainval[target_date_col].max().strftime("%Y-%m-%d")],
        "test_observation_range": [
            test[DATE_COL].min().strftime("%Y-%m-%d"), test[DATE_COL].max().strftime("%Y-%m-%d")],
        "model_file": str(model_path.relative_to(ROOT)),
        "test_predictions_file": str(preds_path_name(horizon)),
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT_DIR / f"rf_summary_t{horizon}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return preds, summary

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Khong tim thay {INPUT}. Can tich hop du lieu TV3 truoc khi chay."
        )
    df = pd.read_csv(INPUT)
    required = {DATE_COL, "price", "target_1", "split_1", "target_date_1",
                "target_5", "split_5", "target_date_5"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Thieu cot bat buoc trong brent_features.csv: {missing}")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    all_summaries = []
    all_predictions = []
    for h in (1, 5):
        preds, summary = tune_and_predict(df, h)
        all_predictions.append(preds)
        all_summaries.append(summary)
        pb = summary["purged_boundary_samples"]
        print(f"\n=== Horizon {h} phien ===")
        print(f"Rows train/val/test: {summary['train_rows']}/{summary['validation_rows']}/{summary['test_rows']}")
        print(f"Purge Train/Val: {pb['train_val']} | Val/Test: {pb['val_test']}")
        print(f"Best params: {summary['best_params']}")
        print(f"Validation MAE: {summary['best_validation_MAE']:.4f}")
        print(f"Validation RMSE: {summary['best_validation_RMSE']:.4f}")
        print(f"Model: {summary['model_file']} | Metadata: {summary['metadata_file']}")
        print(f"Test predictions: {summary['test_predictions_file']}")
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(RESULT_DIR / "rf_predictions_test.csv", index=False)
    (RESULT_DIR / "rf_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nDa xong. Chi so Test khong duoc dung de chon tham so; TV8 se danh gia tren cac file predictions.")

if __name__ == "__main__":
    main()
