import numpy as np
import pandas as pd
import streamlit as st
from src.core.config import OFFICIAL_HORIZONS
from src.services.data_service import get_clean_data
from src.services.forecast_service import (
    get_available_joblib_models,
    get_all_latest_forecasts,
)
from src.ui.components import (
    render_section_header,
    render_bento_kpis,
    render_genesis_table,
    render_editorial_note,
)
from src.ui.charts import plot_bento_october_forecast


def _validated_forecast_rows(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    required = {
        "model", "horizon", "observation_date", "last_price",
        "predicted_price", "absolute_change", "percent_change",
    }
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Backend không trả DataFrame dự báo")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột dự báo: {sorted(missing)}")
    expected = {(model, horizon) for model in models for horizon in OFFICIAL_HORIZONS}
    actual = set(frame[["model", "horizon"]].itertuples(index=False, name=None))
    if len(frame) != len(expected) or actual != expected:
        raise ValueError("Dự báo thiếu, trùng hoặc sai tổ hợp mô hình/horizon 1–5")
    if frame["observation_date"].isna().any() or frame["observation_date"].nunique() != 1:
        raise ValueError("Ngày quan sát của các dự báo không thống nhất")
    numeric = ["last_price", "predicted_price", "absolute_change", "percent_change"]
    if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in numeric):
        raise ValueError("Giá hoặc mức thay đổi dự báo không phải số")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Giá hoặc mức thay đổi dự báo không hợp lệ")
    if frame["last_price"].nunique() != 1 or not frame["last_price"].gt(0).all():
        raise ValueError("Giá Brent tại ngày quan sát không thống nhất hoặc không hợp lệ")
    return frame


def _build_forecast_table(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        by_horizon = frame.loc[frame["model"].eq(model)].set_index("horizon")
        first = by_horizon.loc[1]
        row = {
            "Mô hình": model,
            "Ngày quan sát gốc": first["observation_date"],
            "Giá gốc ($)": f"${first['last_price']:.2f}",
        }
        for horizon in OFFICIAL_HORIZONS:
            forecast = by_horizon.loc[horizon]
            row[f"Dự báo t+{horizon} ($)"] = f"${forecast['predicted_price']:.2f}"
            row[f"Biến động t+{horizon}"] = (
                f"{forecast['absolute_change']:+.2f} USD ({forecast['percent_change']:+.2f}%)"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_forecast_view():
    try:
        clean_df, has_clean = get_clean_data()
    except Exception as exc:
        st.error(f"Không thể đọc dữ liệu Brent để thực hiện dự báo: {exc}")
        return
    if not has_clean or not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
        st.error("Chưa tìm thấy dữ liệu lịch sử để thực hiện dự báo.")
        return
    try:
        required = {"observation_date", "price"}
        missing = required.difference(clean_df.columns)
        if missing:
            raise ValueError(f"Dữ liệu Brent thiếu cột bắt buộc: {sorted(missing)}")
        dates = pd.to_datetime(clean_df["observation_date"], errors="raise")
        prices = pd.to_numeric(clean_df["price"], errors="raise")
        if dates.isna().any() or dates.duplicated().any() or prices.isna().any():
            raise ValueError("Dữ liệu Brent có ngày hoặc giá bị thiếu hay trùng lặp.")
        if not np.isfinite(prices.to_numpy(dtype=float)).all() or not prices.gt(0).all():
            raise ValueError("Giá Brent phải là số hữu hạn và dương.")
    except Exception as exc:
        st.error(f"Dữ liệu Brent không hợp lệ để thực hiện dự báo: {exc}")
        return

    try:
        avail_models = get_available_joblib_models()
    except Exception as exc:
        st.error(f"Không kiểm tra được các model dự báo chính thức: {exc}")
        return

    render_section_header(
        overline="02 / MULTI-HORIZON PRICE FORECAST",
        title="Dự báo Giá dầu Brent cho 5 phiên quan sát hợp lệ tiếp theo",
        description="Tổng hợp dự báo trực tiếp t+1 đến t+5 từ 4 phương pháp, dựa trên phiên Brent cuối cùng trong dữ liệu repository."
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.3, 1.5, 1.1])
        focus_model = c1.selectbox(
            "Mô hình tiêu điểm trên thẻ KPI",
            options=avail_models,
            index=avail_models.index("Random Forest") if "Random Forest" in avail_models else 0,
        )
        selected_models = c2.multiselect(
            "Mô hình hiển thị trên biểu đồ & bảng đối chiếu",
            options=avail_models,
            default=avail_models,
        )
        hist_window = c3.slider(
            "Số phiên lịch sử gần nhất hiển thị",
            min_value=15,
            max_value=60,
            value=25,
            step=5,
        )

    if not selected_models:
        st.warning("Vui lòng chọn ít nhất một mô hình để hiển thị dự báo.")
        return

    try:
        fc_df = _validated_forecast_rows(get_all_latest_forecasts(selected_models), selected_models)
    except Exception as exc:
        st.error(f"Không có đủ dự báo chính thức cho các mô hình đã chọn: {exc}")
        return

    if focus_model not in selected_models:
        st.info(f"Mô hình tiêu điểm chưa được chọn để hiển thị; KPI dùng {selected_models[0]}.")
        focus_model = selected_models[0]
    focus_sub = fc_df[fc_df["model"] == focus_model].set_index("horizon")

    last_obs_date = str(fc_df["observation_date"].iloc[0])
    last_price = float(fc_df["last_price"].iloc[0])

    row_t1 = focus_sub.loc[1]
    row_t5 = focus_sub.loc[5]

    h1_all = fc_df[fc_df["horizon"] == 1]["predicted_price"]
    h5_all = fc_df[fc_df["horizon"] == 5]["predicted_price"]
    all_forecasts = fc_df["predicted_price"]
    chart_df = fc_df.pivot(index="horizon", columns="model", values="predicted_price").reset_index()

    render_bento_kpis([
        {
            "overline": "Giá khóa sổ phiên gần nhất",
            "badge": last_obs_date,
            "value": f"${last_price:.2f}",
            "meta": "Mốc giá cơ sở trước kỳ dự báo",
            "tone": "neutral",
        },
        {
            "overline": f"Dự báo kỳ hạn t+1 ({focus_model})",
            "badge": f"{row_t1['absolute_change']:+.2f} USD ({row_t1['percent_change']:+.2f}%)",
            "value": f"${row_t1['predicted_price']:.2f}",
            "meta": "Sau 1 phiên giao dịch hợp lệ tiếp theo",
            "tone": "success" if row_t1["absolute_change"] >= 0 else "warning",
        },
        {
            "overline": f"Dự báo kỳ hạn t+5 ({focus_model})",
            "badge": f"{row_t5['absolute_change']:+.2f} USD ({row_t5['percent_change']:+.2f}%)",
            "value": f"${row_t5['predicted_price']:.2f}",
            "meta": "Sau 5 phiên giao dịch hợp lệ tiếp theo",
            "tone": "success" if row_t5["absolute_change"] >= 0 else "warning",
        },
        {
            "overline": "Đồng thuận liên mô hình (t+1 → t+5)",
            "badge": f"{len(fc_df['model'].unique())} mô hình",
            "value": f"${h1_all.mean():.2f} → ${h5_all.mean():.2f}",
            "meta": f"Dải dự báo 5 phiên: ${all_forecasts.min():.2f} – ${all_forecasts.max():.2f}",
            "tone": "indigo",
        },
    ])

    with st.container(border=True):
        st.plotly_chart(
            plot_bento_october_forecast(
                hist_df=clean_df.tail(hist_window),
                oct_df=chart_df,
                selected_models=selected_models,
            ),
            use_container_width=True,
        )
        render_editorial_note(
            "Phương pháp dự báo đa kỳ hạn",
            f"Các mô hình dùng 29 đặc trưng kỹ thuật tại phiên quan sát {last_obs_date} để dự báo trực tiếp t+1 đến t+5 phiên Brent hợp lệ tiếp theo; không xác định trước ngày mục tiêu."
        )

    with st.container(border=True):
        st.markdown("#### Bảng đối chiếu chi tiết kết quả dự báo theo từng mô hình")
        st.caption(f"Mốc quan sát cơ sở: {last_obs_date} · Giá đóng cửa thực tế: ${last_price:.2f}/thùng.")

        render_genesis_table(_build_forecast_table(fc_df, selected_models))
