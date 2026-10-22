"""Create chronologically separated forecasts for the supported horizons."""

import pandas as pd

from .config import FORECAST_HORIZONS, RESULT_COLUMNS, SPLIT_BOUNDARIES
from .features import build_features, feature_columns
from .train import fingerprint_data, load_forecasting_artifact


def split_for_date(value) -> str | None:
    """Assign a target date to the requested train/validation/test window."""
    date = pd.Timestamp(value)
    if date <= pd.Timestamp(SPLIT_BOUNDARIES["train_end"]):
        return "train"
    if pd.Timestamp(SPLIT_BOUNDARIES["validation_start"]) <= date <= pd.Timestamp(
        SPLIT_BOUNDARIES["validation_end"]
    ):
        return "validation"
    if pd.Timestamp(SPLIT_BOUNDARIES["test_start"]) <= date <= pd.Timestamp(
        SPLIT_BOUNDARIES["test_end"]
    ):
        return "test"
    return None


def build_forecasts(features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Evaluate validation using train fit, then test using train+validation fit.

    Model settings are fixed in ``train.py``. Validation is reported as a
    chronological holdout; the test window is never used to fit or tune.
    """
    from .train import fit_models

    records = []
    for horizon in FORECAST_HORIZONS:
        target_date_col = f"target_date_{horizon}"
        target_col = f"target_{horizon}"
        frame = features.copy()
        frame = frame.dropna(subset=[target_col, target_date_col]).copy()
        frame["split"] = frame[target_date_col].map(split_for_date)

        for evaluated_split in ("validation", "test"):
            fit_splits = ["train"] if evaluated_split == "validation" else ["train", "validation"]
            fit_rows = frame[frame["split"].isin(fit_splits)]
            eval_rows = frame[frame["split"] == evaluated_split]
            if fit_rows.empty or eval_rows.empty:
                continue

            fitted = fit_models(fit_rows[feature_names], fit_rows[target_col])
            predictions = {
                "Naive Persistence": eval_rows["actual"].to_numpy(),
                **{
                    name: model.predict(eval_rows[feature_names])
                    for name, model in fitted.items()
                },
            }
            for model_name, values in predictions.items():
                for (_, row), predicted in zip(eval_rows.iterrows(), values):
                    records.append(
                        {
                            "model": model_name,
                            "horizon": horizon,
                            "origin_date": row["origin_date"],
                            "target_date": row[target_date_col],
                            "actual": float(row[target_col]),
                            # Tree ensembles can vary by sub-picounit roundoff
                            # across parallel reductions; cents-level inputs
                            # do not justify preserving that numeric noise.
                            "predicted": round(float(predicted), 10),
                            "split": evaluated_split,
                        }
                    )

    return pd.DataFrame(records, columns=RESULT_COLUMNS)


def predict_latest(
    data: pd.DataFrame,
    model_name: str,
    horizon: int,
    artifact_path=None,
) -> dict:
    """Infer a future price from the latest known observation and saved model."""
    from .config import MODEL_ARTIFACT_PATH

    if horizon not in FORECAST_HORIZONS:
        raise ValueError(f"Horizon phải là một trong {FORECAST_HORIZONS}.")
    if model_name not in ("Ridge Regression", "Random Forest", "XGBoost"):
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")

    expected_features = feature_columns()
    bundle = load_forecasting_artifact(artifact_path or MODEL_ARTIFACT_PATH)
    if bundle["data_sha256"] != fingerprint_data(data):
        raise ValueError(
            "Artifact không được huấn luyện trên bản dữ liệu đang chọn. "
            "Hãy cập nhật/huấn luyện lại mô hình trước khi suy luận."
        )

    features = build_features(data)
    latest = features.iloc[-1]
    artifact = bundle["models"].get(f"{model_name}|{horizon}")
    if artifact is None:
        raise ValueError(f"Không tìm thấy artifact {model_name}, horizon {horizon}.")
    metadata = artifact["metadata"]
    if metadata["feature_names"] != expected_features or metadata["horizon"] != horizon:
        raise ValueError("Metadata artifact không khớp đặc trưng hoặc horizon yêu cầu.")

    latest_price = float(latest["actual"])
    inference_row = latest.loc[expected_features].astype("float64").to_frame().T
    predicted_price = float(artifact["estimator"].predict(inference_row)[0])
    change_percent = (predicted_price - latest_price) / latest_price * 100
    return {
        "model": model_name,
        "horizon": horizon,
        "latest_data_date": pd.Timestamp(latest["origin_date"]).strftime("%Y-%m-%d"),
        "latest_observed_price": latest_price,
        "predicted_price": predicted_price,
        "change_percent": change_percent,
        "direction": "Tăng" if change_percent > 0 else "Giảm" if change_percent < 0 else "Không đổi",
        "forecast_label": f"Sau {horizon} phiên quan sát hợp lệ tiếp theo",
        "model_version": metadata["model_version"],
        "training_target_end_date": metadata["training_target_end_date"],
    }
