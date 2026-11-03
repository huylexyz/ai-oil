"""TV4 - Naive Forecast & Ridge Regression cho Brent 1 va 5 phien.

Quy trinh Ridge:
- Validation predictions luon dung model chi fit tren Train (khong fit Validation).
- Test predictions dung model cuoi fit tren Train + Validation hop le.
- Horizon 5 tuning: loai cac mau Train co target_date_5 sau ngay quan sat dau tien
  cua Validation (du kien 4 mau theo bo du lieu TV3).
- Horizon 5 final fit: loai cac mau Validation co target_date_5 sau ngay quan sat
  dau tien cua Test (du kien 4 mau theo bo du lieu TV3).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "processed" / "brent_features.csv"
MODEL_DIR = BASE_DIR / "models"
PREDICTION_DIR = BASE_DIR / "predictions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "price", "lag_1", "lag_2", "lag_3", "lag_5", "lag_10", "lag_20",
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_1_lag1", "ret_1_lag2",
    "ret_1_lag3", "ma_5", "price_ma_5_ratio", "ma_10",
    "price_ma_10_ratio", "ma_20", "price_ma_20_ratio", "ma_50",
    "price_ma_50_ratio", "ma_5_20_ratio", "vol_5", "vol_10", "vol_20",
    "range_20", "dayofweek", "month",
]
ALPHA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
EXPECTED_BOUNDARY_ROWS = 4


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Khong tim thay du lieu: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    required = FEATURES + [
        "observation_date", "target_1", "target_date_1", "split_1",
        "target_5", "target_date_5", "split_5",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Thieu cot trong brent_features.csv: {missing}")
    for col in ["observation_date", "target_date_1", "target_date_5"]:
        data[col] = pd.to_datetime(data[col])
    return data


def select_alpha(train: pd.DataFrame, val: pd.DataFrame, target_col: str,
                 horizon: int) -> tuple[float, float, Pipeline]:
    if train.empty or val.empty:
        raise ValueError(f"Horizon {horizon}: Train hoac Validation rong.")
    best_alpha, best_mae, best_model = None, float("inf"), None
    for alpha in ALPHA_VALUES:
        candidate = make_pipeline(alpha)
        candidate.fit(train[FEATURES], train[target_col])
        prediction = candidate.predict(val[FEATURES])
        mae = mean_absolute_error(val[target_col], prediction)
        print(f"Horizon {horizon} | alpha={alpha:g} | Validation MAE={mae:.4f}")
        if mae < best_mae:
            best_alpha, best_mae, best_model = alpha, mae, candidate
    return best_alpha, best_mae, best_model


def export_staged_predictions(tuning_model: Pipeline, final_model: Pipeline,
                              data: pd.DataFrame, horizon: int,
                              split_col: str, target_col: str,
                              target_date_col: str) -> None:
    """Train/Validation dung tuning model; Test dung final model."""
    output = data.copy()
    output["predicted"] = float("nan")
    non_test = output[split_col].isin(["train", "val"])
    is_test = output[split_col].eq("test")
    if non_test.any():
        output.loc[non_test, "predicted"] = tuning_model.predict(
            output.loc[non_test, FEATURES]
        )
    if is_test.any():
        output.loc[is_test, "predicted"] = final_model.predict(
            output.loc[is_test, FEATURES]
        )

    result = pd.DataFrame({
        "observation_date": output["observation_date"].dt.strftime("%Y-%m-%d"),
        "target_date": output[target_date_col].dt.strftime("%Y-%m-%d"),
        "horizon": horizon,
        "actual": output[target_col],
        "predicted": output["predicted"],
        "model": "Ridge",
        "split": output[split_col],
    })
    path = PREDICTION_DIR / f"ridge_horizon_{horizon}.csv"
    result.to_csv(path, index=False)
    print(f"Da xuat {path} ({len(result)} dong)")


def run_horizon_1(df: pd.DataFrame) -> None:
    data = df[df["split_1"].isin(["train", "val", "test"])
              & df["target_1"].notna()
              & df["target_date_1"].notna()].copy()
    train = data[data["split_1"] == "train"].copy()
    val = data[data["split_1"] == "val"].copy()
    test = data[data["split_1"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, Validation hoac Test horizon 1 dang rong.")

    best_alpha, best_mae, tuning_model = select_alpha(
        train, val, "target_1", 1
    )

    # Theo yeu cau nhom: sau khi chon alpha, fit model cuoi tren Train + Validation.
    final_train = pd.concat([train, val], ignore_index=True)
    final_model = make_pipeline(best_alpha)
    final_model.fit(final_train[FEATURES], final_train["target_1"])

    joblib.dump({
        "model": final_model,
        "features": FEATURES,
        "horizon": 1,
        "alpha": best_alpha,
        "validation_mae": best_mae,
        "training_split": "train + validation",
        "training_rows": len(final_train),
    }, MODEL_DIR / "ridge_horizon_1.joblib")

    export_staged_predictions(
        tuning_model, final_model, data, 1,
        "split_1", "target_1", "target_date_1"
    )
    print(
        f"Horizon 1: alpha={best_alpha:g}; Validation MAE={best_mae:.4f}; "
        f"final training rows={len(final_train)}"
    )


def run_horizon_5(df: pd.DataFrame) -> None:
    data = df[df["split_5"].isin(["train", "val", "test"])
              & df["target_5"].notna()
              & df["target_date_5"].notna()].copy()
    train_all = data[data["split_5"] == "train"].copy()
    val_all = data[data["split_5"] == "val"].copy()
    test = data[data["split_5"] == "test"].copy()
    if train_all.empty or val_all.empty or test.empty:
        raise ValueError("Train, Validation hoac Test horizon 5 dang rong.")

    # TUNING: chi giu mau Train co nhan da xuat hien tai ngay quan sat
    # dau tien cua Validation. Khong loai bat ky dong Validation nao.
    first_val_observation = val_all["observation_date"].min()
    train_tuning = train_all[
        train_all["target_date_5"] <= first_val_observation
    ].copy()
    removed_train = train_all[
        train_all["target_date_5"] > first_val_observation
    ].copy()

    if len(removed_train) != EXPECTED_BOUNDARY_ROWS:
        raise ValueError(
            "Ranh gioi Train/Validation horizon 5 khong dung 4 mau: "
            f"tim thay {len(removed_train)}. "
            f"Ngay quan sat Validation dau tien: {first_val_observation.date()}. "
            "Kiem tra lai du lieu TV3 va quy tac chia split."
        )

    best_alpha, best_mae, tuning_model = select_alpha(
        train_tuning, val_all, "target_5", 5
    )

    # FINAL FIT: Train + Validation, nhung chi giu nhan da xuat hien
    # tai ngay quan sat Test dau tien. Dieu nay loai 4 mau Validation cuoi.
    first_test_observation = test["observation_date"].min()
    val_final = val_all[
        val_all["target_date_5"] <= first_test_observation
    ].copy()
    removed_val = val_all[
        val_all["target_date_5"] > first_test_observation
    ].copy()

    if len(removed_val) != EXPECTED_BOUNDARY_ROWS:
        raise ValueError(
            "Ranh gioi Validation/Test horizon 5 khong dung 4 mau: "
            f"tim thay {len(removed_val)}. "
            f"Ngay quan sat Test dau tien: {first_test_observation.date()}. "
            "Kiem tra lai du lieu TV3 va quy tac chia split."
        )

    final_train = pd.concat([train_all, val_final], ignore_index=True)
    final_model = make_pipeline(best_alpha)
    final_model.fit(final_train[FEATURES], final_train["target_5"])

    joblib.dump({
        "model": final_model,
        "horizon": 5,
        "features": FEATURES,
        "alpha": best_alpha,
        "validation_mae": best_mae,
        "training_split": "train + validation (labels available by first test observation)",
        "training_rows": len(final_train),
        "tuning_train_rows": len(train_tuning),
        "validation_rows_used_for_final_fit": len(val_final),
        "excluded_train_boundary_rows_for_tuning": len(removed_train),
        "excluded_validation_boundary_rows_for_final_fit": len(removed_val),
        "first_validation_observation_date": first_val_observation.strftime("%Y-%m-%d"),
        "first_test_observation_date": first_test_observation.strftime("%Y-%m-%d"),
    }, MODEL_DIR / "ridge_horizon_5.joblib")

    print("\nHorizon 5 - kiem tra ranh gioi")
    print(
        f"Tuning: loai {len(removed_train)} mau Train cuoi: "
        f"{removed_train['observation_date'].dt.strftime('%Y-%m-%d').tolist()}"
    )
    print(
        f"Final fit: loai {len(removed_val)} mau Validation cuoi: "
        f"{removed_val['observation_date'].dt.strftime('%Y-%m-%d').tolist()}"
    )
    print(
        f"Train tuning: {len(train_tuning)} | Validation tuning: {len(val_all)} "
        f"| Final training: {len(final_train)} | Test: {len(test)}"
    )

    export_staged_predictions(
        tuning_model, final_model, data, 5,
        "split_5", "target_5", "target_date_5"
    )
    print(
        f"Horizon 5: alpha={best_alpha:g}; Validation MAE={best_mae:.4f}; "
        f"final training rows={len(final_train)}"
    )


def main() -> None:
    df = load_data()
    run_horizon_1(df)
    print("\n" + "=" * 60 + "\n")
    run_horizon_5(df)
    print("\nHOAN TAT RIDGE REGRESSION TV4")


if __name__ == "__main__":
    main()
