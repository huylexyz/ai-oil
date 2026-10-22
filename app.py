"""Streamlit dashboard for the Brent forecasting experiment."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DEFAULT_DATA_PATH, FORECASTS_PATH, METRICS_PATH
from src.data_loader import DATE_COLUMN, PRICE_COLUMN, load_brent_data
from src.experiment import run_experiment

st.set_page_config(page_title="Dự báo giá dầu Brent", layout="wide")
st.title("Dự báo giá dầu Brent")
st.caption("Thực nghiệm chuỗi thời gian 2015–09/2026 · Horizon tính theo phiên quan sát hợp lệ")

if not DEFAULT_DATA_PATH.is_file():
    st.error(f"Không tìm thấy tệp dữ liệu: {DEFAULT_DATA_PATH}")
    st.stop()

history = load_brent_data()
st.subheader("Dữ liệu Brent")
history_figure = px.line(
    history,
    x=DATE_COLUMN,
    y=PRICE_COLUMN,
    labels={DATE_COLUMN: "Ngày", PRICE_COLUMN: "USD/thùng"},
    title="Giá dầu Brent theo ngày quan sát",
)
st.plotly_chart(history_figure, use_container_width=True)
st.caption(
    f"{len(history):,} quan sát hợp lệ · "
    f"{history[DATE_COLUMN].min():%Y-%m-%d} đến {history[DATE_COLUMN].max():%Y-%m-%d}"
)

if not METRICS_PATH.is_file() or not FORECASTS_PATH.is_file():
    st.info("Chưa có kết quả đã lưu. Chạy `python -m src.experiment` để tạo kết quả thực nghiệm.")
else:
    metrics = pd.read_csv(METRICS_PATH)
    forecasts = pd.read_csv(FORECASTS_PATH, parse_dates=["origin_date", "target_date"])
    quality_path = METRICS_PATH.parent / "data_quality.json"
    if quality_path.is_file():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        st.success(
            f"Đã xác nhận {quality['valid_observations']:,} quan sát hợp lệ "
            f"trên {quality['raw_rows']:,} dòng CSV; "
            f"{quality['missing_or_non_numeric_prices']} giá trị thiếu được bỏ qua."
        )

    st.subheader("Đánh giá trên dữ liệu chưa dùng để fit")
    split_choice = st.radio("Tập đánh giá", ["test", "validation"], horizontal=True)
    selected_horizon = st.selectbox("Horizon đánh giá (phiên)", [1, 5])
    selected_metrics = metrics[
        (metrics["split"] == split_choice)
        & (metrics["horizon"] == selected_horizon)
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
        st.dataframe(display_metrics, hide_index=True, use_container_width=True)
    st.caption(
        "Directional Accuracy so sánh dấu của thay đổi dự báo và thực tế so với giá tại ngày gốc. "
        "Naive Persistence giữ nguyên giá nên không có dự báo hướng; DA được để N/A. "
        "DA quanh 50% là tỷ lệ quan sát được trên mẫu này, không đảm bảo năng lực dự báo; "
        "các horizon nhiều phiên có mục tiêu chồng lấn."
    )

    st.subheader("So sánh dự báo")
    col1, col2 = st.columns(2)
    with col2:
        available_models = forecasts["model"].drop_duplicates().tolist()
        selected_model = st.selectbox("Mô hình", available_models)
    with col1:
        st.write(f"Horizon biểu đồ: **{selected_horizon} phiên**")
    comparison = forecasts[
        (forecasts["split"] == split_choice)
        & (forecasts["horizon"] == selected_horizon)
        & (forecasts["model"] == selected_model)
    ].sort_values("target_date")
    if not comparison.empty:
        chart_data = comparison.melt(
            id_vars="target_date",
            value_vars=["actual", "predicted"],
            var_name="series",
            value_name="USD/thùng",
        )
        chart = px.line(
            chart_data,
            x="target_date",
            y="USD/thùng",
            color="series",
            labels={"target_date": "Ngày đích", "series": "Chuỗi"},
            title=f"{selected_model} · dự báo trước {selected_horizon} phiên · {split_choice}",
        )
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(comparison.tail(20), hide_index=True, use_container_width=True)

    st.caption(
        "Quy trình: giá thiếu được bỏ qua; đặc trưng trễ và rolling chỉ nhìn về quá khứ. "
        "Ridge chuẩn hóa trong pipeline fit; mô hình Test chỉ fit trên Train+Validation. "
        "Tham số được cố định trước và không tối ưu trên Test."
    )

if st.button("Chạy lại thực nghiệm", type="secondary"):
    with st.spinner("Đang fit mô hình và tạo dự báo theo thứ tự thời gian..."):
        run_experiment()
    st.success("Đã cập nhật kết quả.")
    st.rerun()
