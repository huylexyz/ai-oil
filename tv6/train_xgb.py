from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "data/processed/brent_features.csv"
OUT_DIR = ROOT / "results/baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5


def metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
    }


def main():
    df = pd.read_csv(DATA_FILE)

    target = f"target_{HORIZON}"
    target_date = f"target_date_{HORIZON}"
    split = f"split_{HORIZON}"

    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df[target_date] = pd.to_datetime(df[target_date])

    features = [
        c for c in df.columns
        if c != "observation_date"
        and not c.startswith(("target_", "split_"))
    ]
    assert len(features) == 29

    train = df[df[split] == "train"].copy()
    val = df[df[split] == "val"].copy()

    # Nhãn huấn luyện phải có sẵn khi bắt đầu dự báo Validation.
    cutoff = val["observation_date"].min()
    train = train[train[target_date] <= cutoff].copy()

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
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

    fit_seconds = time.perf_counter() - start

    predicted = model.predict(X_val)
    xgb_scores = metrics(y_val, predicted)
    naive_scores = metrics(y_val, val["price"])

    summary = {
        "stage": "baseline_validation",
        "horizon": HORIZON,
        "train_rows": len(train),
        "validation_rows": len(val),
        "best_n_estimators": int(model.best_iteration + 1),
        "fit_seconds": round(fit_seconds, 3),
        "xgboost_validation": xgb_scores,
        "naive_validation": naive_scores,
        "params": model.get_params(),
        "features": features,
    }

    predictions = pd.DataFrame({
        "observation_date": val["observation_date"],
        "target_date": val[target_date],
        "horizon": HORIZON,
        "actual": y_val,
        "predicted": predicted,
        "model": "XGBoost",
        "split": "val",
    })

    predictions.to_csv(
        OUT_DIR / f"xgb_validation_h{HORIZON}.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    joblib.dump(
        {
            "model": model,
            "features": features,
            "horizon": HORIZON,
        },
        OUT_DIR / f"xgb_baseline_h{HORIZON}.joblib",
    )

    (OUT_DIR / f"xgb_baseline_h{HORIZON}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"HORIZON: {HORIZON}")
    print(f"Train: {len(train)} | Validation: {len(val)}")
    print(f"So cay duoc chon: {model.best_iteration + 1}")
    print(f"Thoi gian fit: {fit_seconds:.2f} giay")
    print()
    print("Validation:")
    print(
        f"Naive   | MAE={naive_scores['MAE']:.4f}"
        f" | RMSE={naive_scores['RMSE']:.4f}"
    )
    print(
        f"XGBoost | MAE={xgb_scores['MAE']:.4f}"
        f" | RMSE={xgb_scores['RMSE']:.4f}"
    )
    print(f"\nDa luu ket qua tai: {OUT_DIR}")


if __name__ == "__main__":
    main()