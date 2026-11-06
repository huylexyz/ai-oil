"""Forecast from the latest official Brent observation and approved artifacts."""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from .config import PROCESSED_DATA_DIR, PROJECT_ROOT
from .features import FEATURE_COLUMNS, PRICE, build_features, feature_columns

CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "brent_clean.csv"
MODELS = ("Naive", "Ridge", "Random Forest", "XGBoost")
HORIZONS = (1, 2, 3, 4, 5)
MODEL_FILES = {
    (model, horizon): PROJECT_ROOT / pattern.format(h=horizon)
    for model, pattern in (
        ("Ridge", "tv4/models/ridge_horizon_{h}.joblib"),
        ("Random Forest", "tv5/models/rf_model_t{h}.joblib"),
        ("XGBoost", "tv6/models/xgb_horizon_{h}.joblib"),
    )
    for horizon in HORIZONS
}
RF_METADATA_FILES = {
    horizon: PROJECT_ROOT / f"tv5/models/rf_model_t{horizon}_metadata.json"
    for horizon in HORIZONS
}


def _latest_input() -> tuple[pd.Timestamp, float, pd.DataFrame]:
    if not CLEAN_DATA_FILE.is_file():
        raise FileNotFoundError(f"Thiếu dữ liệu Brent TV2: {CLEAN_DATA_FILE}")
    raw = pd.read_csv(CLEAN_DATA_FILE)
    missing = {"observation_date", PRICE}.difference(raw.columns)
    if missing:
        raise ValueError(f"Dữ liệu Brent thiếu cột {sorted(missing)}")
    if raw.empty:
        raise ValueError("Dữ liệu Brent rỗng")
    raw = raw.loc[:, ["observation_date", PRICE]].copy()
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="raise")
    raw[PRICE] = pd.to_numeric(raw[PRICE], errors="raise")
    if raw.isna().any().any() or not np.isfinite(raw[PRICE].to_numpy(dtype=float)).all():
        raise ValueError("Ngày hoặc giá Brent đầu vào không hợp lệ")
    if raw["observation_date"].duplicated().any():
        raise ValueError("Dữ liệu Brent có ngày quan sát trùng")
    if not raw[PRICE].gt(0).all():
        raise ValueError("Giá Brent phải là số dương")

    latest_date = raw["observation_date"].max()
    features = build_features(raw)
    if features.empty or features.iloc[-1]["observation_date"] != latest_date:
        raise ValueError("Không đủ dữ liệu lịch sử để tạo 29 đặc trưng cho phiên mới nhất")
    names = feature_columns(features)
    actual_order = [column for column in features.columns if column in FEATURE_COLUMNS]
    if len(names) != 29 or actual_order != names:
        raise ValueError("Sai tên hoặc thứ tự 29 đặc trưng TV3")
    latest = features.iloc[[-1]][names]
    if latest.isna().any().any() or not np.isfinite(latest.to_numpy(dtype=float)).all():
        raise ValueError("Đặc trưng của phiên mới nhất không hợp lệ")
    last_price = float(features.iloc[-1]["price"])
    return latest_date, last_price, latest


def _load_model(model: str, horizon: int, expected_features: list[str]):
    path = MODEL_FILES[(model, horizon)]
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu model chính thức: {path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            artifact = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Không nạp được model tương thích {path}: {exc}") from exc

    if model == "Random Forest":
        metadata_path = RF_METADATA_FILES[horizon]
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Thiếu metadata đặc trưng: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact_features = metadata.get("features")
        if metadata.get("horizon") != horizon:
            raise ValueError(f"Metadata Random Forest sai horizon: {metadata_path}")
        estimator = artifact
    else:
        if not isinstance(artifact, dict) or artifact.get("horizon") != horizon:
            raise ValueError(f"Artifact sai định dạng hoặc horizon: {path}")
        artifact_features = artifact.get("features")
        estimator = artifact.get("model")

    if artifact_features != expected_features or len(expected_features) != 29:
        raise ValueError(f"Model sai tên hoặc thứ tự 29 đặc trưng: {path}")
    if getattr(estimator, "n_features_in_", None) != 29:
        raise ValueError(f"Model không nhận đúng 29 đặc trưng: {path}")
    names_in_model = getattr(estimator, "feature_names_in_", None)
    if names_in_model is None or list(names_in_model) != expected_features:
        raise ValueError(f"Model sai tên hoặc thứ tự đặc trưng đầu vào: {path}")
    return estimator


def predict_latest(model: str, horizon: int) -> dict:
    """Predict after the last observed session, without guessing a target date."""
    if model not in MODELS:
        raise ValueError(f"Model không hợp lệ: {model!r}; chọn {MODELS}")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon not in HORIZONS:
        raise ValueError(f"Horizon không hợp lệ: {horizon!r}; chọn 1–5 phiên")

    observation_date, last_price, inputs = _latest_input()
    if model == "Naive":
        predicted_price = last_price
    else:
        estimator = _load_model(model, horizon, list(inputs.columns))
        try:
            prediction = estimator.predict(inputs)
        except Exception as exc:
            raise RuntimeError(f"Model {model} horizon {horizon} không tương thích với đầu vào: {exc}") from exc
        if len(prediction) != 1:
            raise ValueError(f"Model {model} không trả về đúng một dự báo")
        predicted_price = float(prediction[0])
    if not np.isfinite(predicted_price) or predicted_price <= 0:
        raise ValueError(f"Giá dự báo của {model} không hợp lệ: {predicted_price}")
    absolute_change = predicted_price - last_price
    return {
        "model": model,
        "horizon": int(horizon),
        "observation_date": observation_date.strftime("%Y-%m-%d"),
        "last_price": last_price,
        "predicted_price": predicted_price,
        "absolute_change": absolute_change,
        "percent_change": absolute_change / last_price * 100.0,
    }
