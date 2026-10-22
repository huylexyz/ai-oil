"""Streamlit dashboard for Brent backtesting and latest-data forecasting."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    DEFAULT_DATA_PATH,
    FORECASTS_PATH,
    FORECAST_HORIZONS,
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
)
from src.data_loader import DATE_COLUMN, PRICE_COLUMN, load_current_brent_data
from src.data_update import update_brent_data_and_models
from src.experiment import run_historical_experiment
from src.predict import predict_latest
from src.train import fingerprint_data, load_forecasting_artifact

st.set_page_config(page_title="Dự báo giá dầu Brent", layout="wide")
st.title("Dự báo giá dầu Brent")
st.caption("Dự báo theo 1 hoặc 5 phiên quan sát hợp lệ")

try:
    history, active_data_path = load_current_brent_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Không đọc được dữ liệu Brent: {error}")
    st.stop()

latest_data_date = history[DATE_COLUMN].max().strftime("%Y-%m-%d")
latest_price = float(history.iloc[-1][PRICE_COLUMN])
st.subheader("Dữ liệu giá Brent")
st.caption(
    f"Nguồn đang dùng: `{active_data_path}` · {len(history):,} quan sát hợp lệ · "
    f"dữ liệu mới nhất {latest_data_date} · giá gần nhất {latest_price:.2f} USD/thùng"
)
history_figure = px.line(
    history,
    x=DATE_COLUMN,
    y=PRICE_COLUMN,
    labels={DATE_COLUMN: "Ngày quan sát", PRICE_COLUMN: "USD/thùng"},
    title="Giá dầu Brent theo ngày quan sát",
)
st.plotly_chart(history_figure, width="stretch")

st.header("Dự báo mới từ dữ liệu gần nhất")
st.info(
    f"Dự báo được tính từ ngày dữ liệu cuối ({latest_data_date}), không phải từ ngày hiện tại. "
    "Horizon là số phiên quan sát hợp lệ tiếp theo; ngày giao dịch tương lai chưa được xác định."
)

artifact = None
artifact_matches_data = False
try:
    artifact = load_forecasting_artifact()
    artifact_matches_data = artifact["data_sha256"] == fingerprint_data(history)
except (FileNotFoundError, ValueError, OSError) as error:
    st.warning(f"Chưa có artifact suy luận dùng được: {error}")

if artifact is not None:
    artifact_models = artifact["models"]
    artifact_horizons = sorted(
        {entry["metadata"]["horizon"] for entry in artifact_models.values()}
    )
    artifact_model_names = sorted(
        {entry["metadata"]["model_name"] for entry in artifact_models.values()}
    )
    first_entry = next(iter(artifact_models.values()))["metadata"]
    st.caption(
        f"Artifact: `{MODEL_ARTIFACT_PATH}` · phiên bản {artifact['artifact_version']} · "
        f"dữ liệu huấn luyện cuối {artifact['latest_data_date']} · "
        f"nhãn huấn luyện đến {first_entry['training_target_end_date']}"
    )
    if not artifact_matches_data:
        st.warning("Dữ liệu đã thay đổi so với artifact. Hãy cập nhật và huấn luyện lại trước khi dự báo.")

    forecast_col1, forecast_col2 = st.columns(2)
    with forecast_col1:
        selected_model = st.selectbox("Mô hình dự báo", artifact_model_names)
    with forecast_col2:
        selected_horizon = st.selectbox(
            "Horizon mới (phiên)",
            [h for h in FORECAST_HORIZONS if h in artifact_horizons],
        )
    if st.button("Tạo dự báo mới", type="primary", disabled=not artifact_matches_data):
        try:
            result = predict_latest(history, selected_model, selected_horizon)
            st.subheader(result["forecast_label"])
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Giá gần nhất", f"{result['latest_observed_price']:.2f} USD/thùng")
            metric_col2.metric("Giá dự báo", f"{result['predicted_price']:.2f} USD/thùng")
            metric_col3.metric("Thay đổi dự báo", f"{result['change_percent']:+.2f}%")
            metric_col4.metric("Xu hướng dự báo", result["direction"])
            st.caption(
                f"Mốc dữ liệu: {result['latest_data_date']} · {result['model']} v{result['model_version']} · "
                f"artifact huấn luyện đến {result['training_target_end_date']}. "
                "Đây là giá dự báo, chưa phải giá thực tế đã quan sát."
            )
        except (ValueError, FileNotFoundError, OSError) as error:
            st.error(f"Không thể tạo dự báo: {error}")

st.header("Cập nhật dữ liệu và huấn luyện mô hình")
st.caption(
    "Tải CSV có cột observation_date và DCOILBRENTEU. Dữ liệu được hợp nhất vào bản dẫn xuất "
    "trong data/processed; CSV gốc và các kết quả backtest đã lưu không bị ghi đè."
)
uploaded_csv = st.file_uploader("Chọn CSV Brent từ FRED/EIA", type=["csv"], key="brent_csv")
if st.button("Cập nhật dữ liệu & huấn luyện lại mô hình", disabled=uploaded_csv is None):
    try:
        uploaded_frame = pd.read_csv(uploaded_csv)
        with st.spinner("Đang kiểm tra dữ liệu, hợp nhất và fit các mô hình suy luận..."):
            update_result = update_brent_data_and_models(uploaded_frame)
        if update_result["updated"]:
            st.success(
                f"Đã cập nhật {update_result['added_observations']:,} quan sát mới, "
                f"điều chỉnh {update_result['revised_observations']:,} ngày; "
                f"dữ liệu mới nhất {update_result['latest_data_date']}. "
                f"Bỏ qua {update_result['missing_prices_skipped']:,} giá thiếu."
            )
            st.rerun()
        else:
            st.info(update_result["message"])
    except (ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f"CSV không được áp dụng: {error}")

st.header("Backtesting lịch sử")
st.caption(
    "Bảng và biểu đồ dưới đây là dự báo lịch sử trên các tập Validation/Test; "
    "chúng không phải dự báo mới từ ngày dữ liệu hiện tại."
)
if not METRICS_PATH.is_file() or not FORECASTS_PATH.is_file():
    st.info("Chưa có kết quả backtest. Chạy `python -m src.experiment` để tạo kết quả và artifact ban đầu.")
else:
    metrics = pd.read_csv(METRICS_PATH)
    forecasts = pd.read_csv(FORECASTS_PATH, parse_dates=["origin_date", "target_date"])
    split_choice = st.radio("Tập backtest", ["test", "validation"], horizontal=True)
    historical_horizon = st.selectbox("Horizon backtest (phiên)", [1, 5])
    selected_metrics = metrics[
        (metrics["split"] == split_choice)
        & (metrics["horizon"] == historical_horizon)
    ].copy()
    if not selected_metrics.empty:
        display_metrics = selected_metrics.copy()
        display_metrics["Directional Accuracy"] = display_metrics[
            "Directional Accuracy"
        ].map(
            lambda value: f"{value:.1%}"
            if pd.notna(value)
            else "N/A — persistence predicts no direction"
        )
        for column in ("MAE", "RMSE"):
            display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.3f}")
        st.dataframe(display_metrics, hide_index=True, width="stretch")

    model_options = forecasts["model"].drop_duplicates().tolist()
    selected_historical_model = st.selectbox("Mô hình trên backtest", model_options)
    historical = forecasts[
        (forecasts["split"] == split_choice)
        & (forecasts["horizon"] == historical_horizon)
        & (forecasts["model"] == selected_historical_model)
    ].sort_values("target_date")
    if not historical.empty:
        chart_data = historical.melt(
            id_vars="target_date",
            value_vars=["actual", "predicted"],
            var_name="series",
            value_name="USD/thùng",
        )
        st.plotly_chart(
            px.line(
                chart_data,
                x="target_date",
                y="USD/thùng",
                color="series",
                labels={"target_date": "Ngày đích lịch sử", "series": "Chuỗi"},
                title=f"Backtest · {selected_historical_model} · {historical_horizon} phiên · {split_choice}",
            ),
            width="stretch",
        )
        st.dataframe(historical.tail(20), hide_index=True, width="stretch")

if st.button("Chạy lại backtest lịch sử", type="secondary"):
    with st.spinner("Đang chạy lại đánh giá lịch sử trên CSV gốc..."):
        run_historical_experiment(DEFAULT_DATA_PATH)
    st.success("Đã cập nhật kết quả backtest; artifact suy luận hiện tại được giữ nguyên.")
    st.rerun()
