import numpy as np
import pandas as pd
from src.evaluate import calculate_metrics
from src.predict import predict_latest as backend_predict_latest
from src.core.config import MODELS_LIST, HORIZONS, FEATURE_COLS
from src.services.data_service import load_trained_model, get_features_data


def get_available_joblib_models() -> list[str]:
    avail = ["Naive"]
    for m in ["Ridge", "Random Forest", "XGBoost"]:
        _, _, ok1 = load_trained_model(m, 1)
        _, _, ok5 = load_trained_model(m, 5)
        if ok1 and ok5:
            avail.append(m)
    return avail


def get_latest_official_forecast(model_name: str, horizon: int) -> dict:
    try:
        return backend_predict_latest(model_name, int(horizon))
    except Exception:
        feat_df, ok = get_features_data()
        if not ok or feat_df.empty:
            raise
        last_row = feat_df.iloc[-1]
        last_price = float(last_row["price"])
        obs_date = pd.to_datetime(last_row["observation_date"]).strftime("%Y-%m-%d")
        X_last = feat_df.tail(1)[FEATURE_COLS]
        est, _, ok_m = load_trained_model(model_name, int(horizon))
        if not ok_m or est is None:
            raise RuntimeError(f"Không nạp được mô hình {model_name} (t+{horizon})")
        pred_price = last_price if est == "NAIVE_RULE" else float(est.predict(X_last)[0])
        abs_change = pred_price - last_price
        return {
            "model": model_name,
            "horizon": int(horizon),
            "observation_date": obs_date,
            "last_price": last_price,
            "predicted_price": pred_price,
            "absolute_change": abs_change,
            "percent_change": (abs_change / last_price) * 100.0,
        }


def get_all_latest_forecasts(models: list[str] = None) -> pd.DataFrame:
    target_models = models or get_available_joblib_models()
    rows = []
    for m in target_models:
        for h in HORIZONS:
            try:
                res = get_latest_official_forecast(m, h)
                rows.append(res)
            except Exception:
                continue
    return pd.DataFrame(rows)


def generate_october_week_forecast(*args, **kwargs) -> pd.DataFrame:
    method = kwargs.get("method", "direct")
    for a in args:
        if isinstance(a, str):
            method = a

    avail_models = get_available_joblib_models()
    future_dates = pd.bdate_range(start="2026-10-01", periods=5)
    oct_df = pd.DataFrame({"target_date": future_dates})
    oct_df["Phiên"] = [f"Phiên t+{i}" for i in range(1, 6)]

    for m in avail_models:
        try:
            p1 = float(get_latest_official_forecast(m, 1)["predicted_price"])
            p5 = float(get_latest_official_forecast(m, 5)["predicted_price"])
        except Exception:
            continue

        if "step" in str(method).lower() or "bậc" in str(method).lower():
            vals = [p1, p1, p1, p1, p5]
        else:
            vals = np.linspace(p1, p5, 5).tolist()
        oct_df[m] = np.round(vals, 4)

    return oct_df


def _calc_fallback_da_pct(df_model: pd.DataFrame) -> float:
    """Tính tỷ lệ dự báo đúng xu hướng (%) qua biến thiên liên phiên cho mô hình Naive."""
    if df_model is None or len(df_model) < 2:
        return 50.0
    s = df_model.sort_values("target_date")
    act_diff = np.sign(np.diff(s["actual"].to_numpy(dtype=float)))
    pred_diff = np.sign(np.diff(s["predicted"].to_numpy(dtype=float)))
    return float(np.mean(act_diff == pred_diff) * 100.0)


def compute_evaluation_metrics(df_sub: pd.DataFrame, selected_models: list[str]) -> pd.DataFrame:
    filtered = df_sub[df_sub["model"].isin(selected_models)].copy()
    if "split" in filtered.columns and (filtered["split"] == "test").any():
        filtered = filtered[filtered["split"] == "test"].copy()
    if filtered.empty:
        return pd.DataFrame()

    raw_metrics = calculate_metrics(filtered)
    raw_metrics = raw_metrics.sort_values("rmse", ascending=True).reset_index(drop=True)

    rows = []
    for idx, r in raw_metrics.iterrows():
        m_name = str(r["model"])
        da_val = r["directional_accuracy"]
        if pd.isna(da_val):
            m_df = filtered[filtered["model"] == m_name]
            da_num = round(_calc_fallback_da_pct(m_df), 2)
        else:
            da_num = round(float(da_val), 2)

        rows.append({
            "Hạng": idx + 1,
            "Mô hình": m_name,
            "Số phiên": int(r["n_samples"]),
            "MAE (USD)": round(float(r["mae"]), 4),
            "RMSE (USD)": round(float(r["rmse"]), 4),
            "Directional Accuracy (%)": f"{da_num:.2f}%",
            "_da_num": da_num,
        })
    return pd.DataFrame(rows)