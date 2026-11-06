import pandas as pd
import streamlit as st
from src.core.config import FEATURE_GLOSSARY, METRICS_GLOSSARY, FEATURE_COLS
from src.core.validators import validate_custom_inputs
from src.services.data_service import load_trained_model
from src.ui.components import render_genesis_table

dialog_decorator = getattr(st, "dialog", getattr(st, "experimental_dialog", None))

if dialog_decorator is not None:
    @dialog_decorator("Từ điển 29 Đặc trưng Kỹ thuật & Chỉ số Đánh giá", width="large")
    def open_glossary_modal():
        _render_glossary_body()

    @dialog_decorator("Trình Giả lập Kịch bản Giá & Kiểm thử Bắt lỗi Nhập liệu", width="large")
    def open_simulator_modal(base_row: dict, focus_model: str):
        _render_simulator_body(base_row, focus_model)
else:
    def open_glossary_modal():
        with st.expander("Từ điển 29 Đặc trưng Kỹ thuật & Chỉ số Đánh giá", expanded=True):
            _render_glossary_body()

    def open_simulator_modal(base_row: dict, focus_model: str):
        with st.expander("Trình Giả lập Kịch bản Giá & Kiểm thử Bắt lỗi Nhập liệu", expanded=True):
            _render_simulator_body(base_row, focus_model)


def _render_glossary_body():
    st.markdown("**1. Bảng giải thích 3 chỉ số chấm điểm mô hình (Trên tập Test độc lập)**")
    render_genesis_table(pd.DataFrame(METRICS_GLOSSARY))
    st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
    st.markdown("**2. Bảng giải thích 6 nhóm đặc trưng đầu vào (Tổng cộng 29 biến)**")
    render_genesis_table(pd.DataFrame(FEATURE_GLOSSARY), max_height=320)


def _render_simulator_body(base_row: dict, focus_model: str):
    st.caption(
        "Nhập thông số giả lập để kiểm tra độ nhạy của mô hình `.joblib` và cơ chế chặn dữ liệu bất hợp lệ "
        "(thử nhập giá âm hoặc vượt ngưỡng 300 USD/thùng để kiểm chứng bộ lọc lỗi)."
    )
    c1, c2, c3 = st.columns(3)
    p_in = c1.number_input("Giá giao ngay (price - USD)", value=float(base_row["price"]), step=0.5)
    lag1_in = c2.number_input("Giá phiên trước (lag_1 - USD)", value=float(base_row["lag_1"]), step=0.5)
    ma5_in = c3.number_input("Trung bình 5 phiên (ma_5 - USD)", value=float(base_row["ma_5"]), step=0.5)

    c4, c5, c6 = st.columns(3)
    ma20_in = c4.number_input("Trung bình 20 phiên (ma_20 - USD)", value=float(base_row["ma_20"]), step=0.5)
    vol5_in = c5.number_input("Độ biến động 5 phiên (vol_5)", value=float(base_row["vol_5"]), step=0.002, format="%.4f")
    ret1_pct = c6.number_input("Tỷ suất sinh lời 1 phiên (%)", value=float(base_row["ret_1"] * 100), step=0.25, format="%.2f")

    if st.button("Thực thi kiểm tra & Chạy dự báo", type="primary", use_container_width=True):
        is_ok, errors = validate_custom_inputs(p_in, lag1_in, ma5_in, ma20_in, vol5_in, ret1_pct)
        if not is_ok:
            for err in errors:
                st.error(f"Từ chối dữ liệu đầu vào: {err}")
        else:
            sim = base_row.copy()
            sim["price"] = p_in
            sim["lag_1"] = lag1_in
            sim["ma_5"] = ma5_in
            sim["ma_20"] = ma20_in
            sim["vol_5"] = vol5_in
            sim["ret_1"] = ret1_pct / 100.0
            sim["price_ma_5_ratio"] = p_in / ma5_in - 1.0
            sim["price_ma_20_ratio"] = p_in / ma20_in - 1.0
            sim["ma_5_20_ratio"] = ma5_in / ma20_in - 1.0

            X_sim = pd.DataFrame([sim])[FEATURE_COLS]
            m1, _, _ = load_trained_model(focus_model, 1)
            m5, _, _ = load_trained_model(focus_model, 5)
            pred_1 = p_in if m1 == "NAIVE_RULE" else float(m1.predict(X_sim)[0])
            pred_5 = p_in if m5 == "NAIVE_RULE" else float(m5.predict(X_sim)[0])

            st.success(
                f"Xác thực hợp lệ — Kết quả chạy mô hình {focus_model}: "
                f"Sau 1 phiên (t+1) = ${pred_1:.2f}/thùng ({pred_1 - p_in:+.2f} USD) | "
                f"Sau 5 phiên (t+5) = ${pred_5:.2f}/thùng ({pred_5 - p_in:+.2f} USD)."
            )