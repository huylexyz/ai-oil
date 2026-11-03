from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "data/processed/brent_features.csv"
TUNING_DIR = ROOT / "results/tuning"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"


def finalize_horizon(df, horizon):
    summary_file = TUNING_DIR / f"xgb_tuning_summary_h{horizon}.json"
    tuning = json.loads(summary_file.read_text(encoding="utf-8"))

    features = tuning["features"]
    best = tuning["best_trial"]

    target = f"target_{horizon}"
    target_date = f"target_date_{horizon}"
    split = f"split_{horizon}"

    work = df.copy()
    work["observation_date"] = pd.to_datetime(work["observation_date"])
    work[target_date] = pd.to_datetime(work[target_date])

    trainval = work[work[split].isin(["train", "val"])].copy()
    test = work[work[split] == "test"].copy()

    assert not trainval.empty and not test.empty

    # Dự báo sau khi đã biết giá của phiên observation_date.
    # Chỉ học từ nhãn có sẵn ở lần dự báo Test đầu tiên.
    first_test_origin = test["observation_date"].min()
    before_filter = len(trainval)
    trainval = trainval[
        trainval[target_date] <= first_test_origin
    ].copy()

    assert trainval[target_date].max() <= first_test_origin
    assert trainval[target].notna().all()
    assert test[target].notna().all()
    assert test["observation_date"].is_unique
    assert np.isfinite(
        trainval[features].to_numpy(dtype=float)
    ).all()
    assert np.isfinite(
        test[features].to_numpy(dtype=float)
    ).all()

    # Giữ cấu hình đã chọn; thay early stopping bằng số vòng cố định.
    params = tuning["selected_model_params"].copy()
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = int(best["best_n_estimators"])

    model = XGBRegressor(**params)

    start = time.perf_counter()
    model.fit(
        trainval[features],
        trainval[target],
        verbose=False,
    )
    fit_seconds = time.perf_counter() - start

    predicted = model.predict(test[features])
    assert np.isfinite(predicted).all()

    model_path = MODEL_DIR / f"xgb_horizon_{horizon}.joblib"

    joblib.dump(
        {
            "model": model,
            "features": features,
            "horizon": horizon,
            "stage": "final_trainval",
            "params": model.get_params(),
            "first_test_origin": first_test_origin.strftime("%Y-%m-%d"),
            "training_target_date_max": (
                trainval[target_date].max().strftime("%Y-%m-%d")
            ),
            "xgboost_version": xgboost.__version__,
        },
        model_path,
    )

    # Kiểm tra mô hình lưu ra có tải lại và dự báo giống nhau.
    loaded = joblib.load(model_path)
    reloaded_pred = loaded["model"].predict(
        test[loaded["features"]]
    )
    np.testing.assert_allclose(
        predicted, reloaded_pred, rtol=1e-7, atol=1e-7
    )

    predictions = pd.DataFrame({
        "observation_date": test["observation_date"],
        "target_date": test[target_date],
        "horizon": horizon,
        "actual": test[target],
        "predicted": predicted,
        "model": "XGBoost",
        "split": "test",
    })

    prediction_path = (
        RESULT_DIR / f"xgb_predictions_test_h{horizon}.csv"
    )
    predictions.to_csv(
        prediction_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary = {
        "horizon": horizon,
        "stage": "final_trainval",
        "selection_metric": "validation_RMSE",
        "selected_validation_MAE": best["validation_MAE"],
        "selected_validation_RMSE": best["validation_RMSE"],
        "trainval_rows_before_filter": before_filter,
        "removed_boundary_rows": before_filter - len(trainval),
        "final_training_rows": len(trainval),
        "test_rows": len(test),
        "first_test_origin": first_test_origin.strftime("%Y-%m-%d"),
        "training_target_date_max": (
            trainval[target_date].max().strftime("%Y-%m-%d")
        ),
        "n_estimators": params["n_estimators"],
        "fit_seconds": round(fit_seconds, 3),
        "params": model.get_params(),
        "features": features,
        "model_file": str(model_path.relative_to(ROOT)),
        "predictions_file": str(prediction_path.relative_to(ROOT)),
    }

    print(f"\nHORIZON {horizon}")
    print("Train + Validation ban dau:", before_filter)
    print("Mau loai tai ranh gioi:", before_filter - len(trainval))
    print("So mau huan luyen cuoi:", len(trainval))
    print("So vong boosting:", params["n_estimators"])
    print("So du bao Test:", len(predictions))
    print("Kiem tra luu/tai mo hinh: DAT")
    print("Model:", model_path)
    print("Predictions:", prediction_path)

    return summary, predictions


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    summaries = []
    predictions = []

    for horizon in (1, 5):
        summary, prediction = finalize_horizon(df, horizon)
        summaries.append(summary)
        predictions.append(prediction)

    pd.concat(predictions, ignore_index=True).to_csv(
        RESULT_DIR / "xgb_predictions_test.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    report = {
        "data_file": str(DATA_FILE),
        "data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "horizons": summaries,
    }

    (RESULT_DIR / "xgb_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nHOAN TAT HUAN LUYEN VA XUAT DU BAO TEST")


if __name__ == "__main__":
    main()