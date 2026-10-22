"""Streamlit entry point for the Brent forecasting project."""

import streamlit as st
import plotly.express as px

from src.config import DEFAULT_DATA_PATH, FORECAST_HORIZONS, MODEL_DIR, RESULT_COLUMNS
from src.data_loader import DATE_COLUMN, PRICE_COLUMN, load_brent_data

st.set_page_config(page_title="Dự báo giá dầu Brent", layout="wide")
st.title("Dự báo biến động giá dầu Brent")
st.caption("Nghiên cứu giá dầu Brent 2015–09/2026 · Horizon tính theo phiên giao dịch")

st.subheader("Giá Brent lịch sử")
if DEFAULT_DATA_PATH.is_file():
    history = load_brent_data()
    if history.empty:
        st.warning("Tệp dữ liệu có tồn tại nhưng không có quan sát hợp lệ.")
    else:
        figure = px.line(
            history,
            x=DATE_COLUMN,
            y=PRICE_COLUMN,
            labels={DATE_COLUMN: "Ngày", PRICE_COLUMN: "USD/thùng"},
            title="Giá dầu Brent theo ngày",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            f"{len(history):,} quan sát hợp lệ · "
            f"{history[DATE_COLUMN].min():%Y-%m-%d} đến {history[DATE_COLUMN].max():%Y-%m-%d}"
        )
else:
    st.warning(f"Chưa có tệp dữ liệu: `{DEFAULT_DATA_PATH}`")

st.subheader("Dự báo và đánh giá mô hình")
st.info(
    "Chưa thể chạy dự báo hoặc trình bày MAE, RMSE, Directional Accuracy: "
    "mô-đun huấn luyện/dự báo/đánh giá và artifact mô hình chưa được bàn giao. "
    "Không hiển thị kết quả thực nghiệm giả."
)
st.caption(
    f"Horizon dự kiến: {', '.join(map(str, FORECAST_HORIZONS))} phiên · "
    f"Schema kết quả thống nhất: {', '.join(RESULT_COLUMNS)} · "
    f"Thư mục artifact: `{MODEL_DIR}`"
)
