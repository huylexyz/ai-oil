from pathlib import Path
from itertools import product
import json
import time

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "data/processed/brent_features.csv"
OUT_DIR = ROOT / "results/tuning"


def scores(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
    }


def tune_horizon(df, features, horizon):
    target = f"target_{horizon}"
    target_date = f"target_date_{horizon}"
    split = f"split_{horizon}"

    work = df.copy()
    work["observation_date"] = pd.to_datetime(work["observation_date"])
    work[target_date] = pd.to_datetime(work[target_date])

    train = work[work[split] == "train"].copy()
    val = work[work[split] == "val"].copy()

    # Loại nhãn chưa có tại lần dự báo Validation đầu tiên.
    cutoff = val["observation_date"].min()
    train = train[train[target_date] <= cutoff].copy()

    assert not train.empty and not val.empty
    assert train[target].notna().all()
    assert val[target].notna().all()

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]

    naive = scores(y_val, val["price"])

    # 3 × 2 × 3 = 18 cấu hình, bao gồm cấu hình baseline.
    candidates = list(product(
        [2, 3, 5],       # max_depth
        [0.03, 0.05],   # learning_rate
        [3, 5, 10],     # min_child_weight
    ))

    rows = []
    best_model = None
    best_row = None

    print(f"\nHORIZON {horizon}")
    print(f"Train: {len(train)} | Validation: {len(val)}")

    for trial, (depth, rate, child_weight) in enumerate(candidates, start=1):
        model = XGBRegressor(
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

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        elapsed = time.perf_counter() - start
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
            "fit_seconds": round(elapsed, 3),
        }
        rows.append(row)

        print(
            f"[{trial:02d}/{len(candidates)}] "
            f"depth={depth} lr={rate} child={child_weight} "
            f"| trees={row['best_n_estimators']} "
            f"| MAE={result['MAE']:.4f} "
            f"| RMSE={result['RMSE']:.4f}"
        )

        # Khi bằng điểm, giữ cấu hình xuất hiện trước.
        if best_row is None or result["RMSE"] < best_row["validation_RMSE"]:
            best_row = row.copy()
            best_model = model

    table = pd.DataFrame(rows).sort_values(
        ["validation_RMSE", "trial"]
    )
    table.to_csv(OUT_DIR / f"xgb_tuning_h{horizon}.csv", index=False)

    # Lưu mô hình của giai đoạn chọn tham số, chưa phải mô hình cuối.
    joblib.dump(
        {
            "model": best_model,
            "features": features,
            "horizon": horizon,
            "stage": "validation_tuning",
        },
        OUT_DIR / f"xgb_selected_h{horizon}.joblib",
    )

    best_predictions = pd.DataFrame({
        "observation_date": val["observation_date"],
        "target_date": val[target_date],
        "horizon": horizon,
        "actual": y_val,
        "predicted": best_model.predict(X_val),
        "model": "XGBoost",
        "split": "val",
    })
    best_predictions.to_csv(
        OUT_DIR / f"xgb_selected_validation_h{horizon}.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    summary = {
        "stage": "validation_tuning",
        "horizon": horizon,
        "train_rows": len(train),
        "validation_rows": len(val),
        "selection_metric": "validation_RMSE",
        "best_trial": best_row,
        "selected_model_params": best_model.get_params(),
        "naive_validation": naive,
        "features": features,
    }
    (OUT_DIR / f"xgb_tuning_summary_h{horizon}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nCAU HINH TOT NHAT:")
    print(json.dumps(best_row, indent=2))
    print(f"Naive Validation RMSE: {naive['RMSE']:.4f}")
    print(f"XGBoost Validation RMSE: {best_row['validation_RMSE']:.4f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)

    features = [
        c for c in df.columns
        if c != "observation_date"
        and not c.startswith(("target_", "split_"))
    ]
    assert len(features) == 29

    for horizon in (1, 5):
        tune_horizon(df, features, horizon)

    print(f"\nHOAN TAT. Ket qua: {OUT_DIR}")


if __name__ == "__main__":
    main()