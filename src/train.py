"""Chronological model fitting for the Brent forecasting experiment."""

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import MODEL_ARTIFACT_PATH
from .data_loader import DATE_COLUMN, PRICE_COLUMN
from .features import build_features, feature_columns

MODEL_ARTIFACT_VERSION = "1.0.0"
MODEL_NAMES = ("Ridge Regression", "Random Forest", "XGBoost")


def make_models() -> dict[str, object]:
    """Create fixed, reproducible model configurations (no test tuning)."""
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def fit_models(train_x, train_y) -> dict[str, object]:
    """Fit each model on the supplied training rows."""
    fitted = {}
    for name, model in make_models().items():
        model.fit(train_x, train_y)
        fitted[name] = model
    return fitted


def fingerprint_data(data: pd.DataFrame) -> str:
    """Return a stable hash for the cleaned date/price observations."""
    canonical = data.loc[:, [DATE_COLUMN, PRICE_COLUMN]].copy()
    canonical = canonical.sort_values(DATE_COLUMN).reset_index(drop=True)
    payload = canonical.to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10f",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def train_forecasting_models(
    data: pd.DataFrame, artifact_path: str | Path = MODEL_ARTIFACT_PATH
) -> dict:
    """Fit production inference models on all available labeled history.

    Call only after historical evaluation has completed. Each model/horizon
    estimator is stored separately within one atomically replaced Joblib bundle.
    """
    from .config import FORECAST_HORIZONS

    features = build_features(data)
    names = feature_columns()
    latest_data_date = pd.Timestamp(data[DATE_COLUMN].max()).strftime("%Y-%m-%d")
    bundle = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_data_date": latest_data_date,
        "data_sha256": fingerprint_data(data),
        "feature_names": names,
        "models": {},
    }

    for horizon in FORECAST_HORIZONS:
        target = f"target_{horizon}"
        target_date = f"target_date_{horizon}"
        labeled = features.dropna(subset=[target, target_date]).copy()
        if labeled.empty:
            raise ValueError(f"Nu există đủ nhãn huấn luyện cho horizon {horizon}.")

        fitted = fit_models(labeled[names], labeled[target])
        target_end_date = pd.Timestamp(labeled[target_date].max()).strftime("%Y-%m-%d")
        for model_name, estimator in fitted.items():
            bundle["models"][f"{model_name}|{horizon}"] = {
                "estimator": estimator,
                "metadata": {
                    "model_name": model_name,
                    "model_version": MODEL_ARTIFACT_VERSION,
                    "horizon": horizon,
                    "feature_names": names,
                    "latest_data_date": latest_data_date,
                    "training_target_end_date": target_end_date,
                    "training_rows": int(len(labeled)),
                    "data_sha256": bundle["data_sha256"],
                    "scikit_learn_version": sklearn.__version__,
                    "xgboost_version": xgboost.__version__,
                },
            }

    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{destination.stem}_", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        joblib.dump(bundle, temp_path, compress=3)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return bundle


def load_forecasting_artifact(artifact_path: str | Path = MODEL_ARTIFACT_PATH) -> dict:
    """Load and validate a saved inference bundle."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có artifact mô hình: {path}")
    bundle = joblib.load(path)
    if bundle.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError("Phiên bản artifact mô hình không được hỗ trợ.")
    if bundle.get("feature_names") != feature_columns():
        raise ValueError("Thứ tự đặc trưng của artifact không khớp mã suy luận.")
    from .config import FORECAST_HORIZONS

    expected_models = {
        f"{model_name}|{horizon}"
        for model_name in MODEL_NAMES
        for horizon in FORECAST_HORIZONS
    }
    if set(bundle.get("models", {})) != expected_models:
        raise ValueError("Artifact thiếu hoặc thừa mô hình/horizon được hỗ trợ.")
    for key, entry in bundle["models"].items():
        model_name, raw_horizon = key.rsplit("|", 1)
        metadata = entry.get("metadata", {})
        if (
            metadata.get("model_name") != model_name
            or metadata.get("horizon") != int(raw_horizon)
            or metadata.get("feature_names") != bundle["feature_names"]
            or metadata.get("latest_data_date") != bundle.get("latest_data_date")
            or metadata.get("data_sha256") != bundle.get("data_sha256")
            or "training_target_end_date" not in metadata
        ):
            raise ValueError(f"Metadata artifact không nhất quán: {key}.")
    return bundle
