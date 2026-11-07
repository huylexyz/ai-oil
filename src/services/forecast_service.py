import pandas as pd
from src.evaluate import calculate_metrics
from src.predict import MODEL_FILES, RF_METADATA_FILES, predict_latest as backend_predict_latest
from src.core.config import MODELS_LIST, OFFICIAL_HORIZONS


def get_available_joblib_models() -> list[str]:
    missing = [
        str(MODEL_FILES[(model, horizon)])
        for model in MODELS_LIST if model != "Naive"
        for horizon in OFFICIAL_HORIZONS
        if not MODEL_FILES[(model, horizon)].is_file()
    ]
    missing.extend(
        str(RF_METADATA_FILES[horizon])
        for horizon in OFFICIAL_HORIZONS
        if not RF_METADATA_FILES[horizon].is_file()
    )
    if missing:
        raise FileNotFoundError(f"Thiếu artifact dự báo chính thức: {', '.join(missing)}")
    return list(MODELS_LIST)


def get_latest_official_forecast(model_name: str, horizon: int) -> dict:
    return backend_predict_latest(model_name, horizon)


def get_all_latest_forecasts(models: list[str] = None) -> pd.DataFrame:
    target_models = list(MODELS_LIST) if models is None else models
    if (
        not isinstance(target_models, (list, tuple))
        or not target_models
        or any(not isinstance(model, str) or model not in MODELS_LIST for model in target_models)
        or len(set(target_models)) != len(target_models)
    ):
        raise ValueError(f"Danh sách mô hình không hợp lệ: {target_models!r}")
    rows = []
    for m in target_models:
        for h in OFFICIAL_HORIZONS:
            try:
                res = get_latest_official_forecast(m, h)
            except Exception as exc:
                raise RuntimeError(f"Không hoàn tất dự báo chính thức {m} t+{h}: {exc}") from exc
            required = (
                "model", "horizon", "observation_date", "last_price",
                "predicted_price", "absolute_change", "percent_change",
            )
            if any(key not in res for key in required) or res["model"] != m or res["horizon"] != h:
                raise RuntimeError(f"Backend trả kết quả không hợp lệ cho {m} t+{h}")
            rows.append({key: res[key] for key in required})
    return pd.DataFrame(rows)


def generate_october_week_forecast(method: str = "direct") -> pd.DataFrame:
    # Kept as an adapter for the existing view; the view/chart still need to
    # render session labels instead of assumed future calendar dates.
    if method != "direct":
        raise ValueError("Chỉ hỗ trợ dự báo trực tiếp bằng model chính thức")

    forecasts = get_all_latest_forecasts()
    by_session = forecasts.pivot(index="horizon", columns="model", values="predicted_price")
    by_session = by_session.loc[OFFICIAL_HORIZONS, MODELS_LIST].reset_index()
    by_session.insert(1, "Phiên", [f"Phiên t+{h}" for h in OFFICIAL_HORIZONS])
    by_session.insert(2, "observation_date", forecasts["observation_date"].iloc[0])
    return by_session


def compute_evaluation_metrics(df_sub: pd.DataFrame, selected_models: list[str]) -> pd.DataFrame:
    filtered = df_sub.loc[df_sub["model"].isin(selected_models) & df_sub["split"].eq("test")].copy()
    return calculate_metrics(filtered).sort_values("rmse").reset_index(drop=True)
