import pandas as pd
import streamlit as st
from src.services.data_service import get_clean_data
from src.services.forecast_service import get_available_joblib_models, get_latest_official_forecast
from src.ui.theme import inject_genesis_theme
from src.ui.modals import open_glossary_modal
from src.views.overview_view import render_overview_view
from src.views.forecast_view import render_forecast_view
from src.views.evaluation_view import render_evaluation_view
from src.views.admin_view import render_admin_view

st.set_page_config(
    page_title="BRENT FORECAST — Phân tích & Dự báo Giá dầu Brent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mặc định khóa cố định giao diện tối (Dark Mode)
st.session_state.theme_mode = "dark"

clean_df, has_clean = get_clean_data()
avail_models = get_available_joblib_models()
focus_m = "Random Forest" if "Random Forest" in avail_models else (avail_models[0] if avail_models else "Naive")

if has_clean and len(clean_df) >= 2:
    last_row = clean_df.iloc[-1]
    prev_row = clean_df.iloc[-2]
    spot_price = float(last_row["price"])
    spot_diff = spot_price - float(prev_row["price"])
    spot_pct = (spot_diff / float(prev_row["price"])) * 100.0
    spot_date = pd.to_datetime(last_row["observation_date"]).strftime("%d/%m/%Y")

    recent_252 = clean_df["price"].tail(252)
    low_52w = float(recent_252.min())
    high_52w = float(recent_252.max())
    range_pct = max(4.0, min(96.0, ((spot_price - low_52w) / max(high_52w - low_52w, 1e-6)) * 100.0))

    try:
        pred_t1 = float(get_latest_official_forecast(focus_m, 1)["predicted_price"])
        pred_t5 = float(get_latest_official_forecast(focus_m, 5)["predicted_price"])
    except Exception:
        pred_t1, pred_t5 = spot_price, spot_price
else:
    spot_price, spot_pct, spot_date = 115.91, 1.71, "30/09/2026"
    low_52w, high_52w, range_pct = 59.93, 138.21, 72.0
    pred_t1, pred_t5 = 116.02, 114.10

diff_t1 = pred_t1 - spot_price
diff_t5 = pred_t5 - spot_price

spot_pct_cls = "delta-pos" if spot_pct >= 0 else "delta-neg"
t1_cls = "delta-pos" if diff_t1 >= 0 else "delta-warn"
t5_cls = "delta-pos" if diff_t5 >= 0 else "delta-warn"

t1_status = "Tăng nhẹ" if diff_t1 >= 0 else "Điều chỉnh"
t5_status = "Tăng nhẹ" if diff_t5 >= 0 else "Điều chỉnh"

with st.sidebar:
    st.markdown(f"""
    <div class="sb-opt4-header">
        <div class="sb-opt4-brand">
            <span class="sb-opt4-mark">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--gen-primary);">
                    <polyline points="3 17 9 11 13 15 21 7"></polyline>
                    <circle cx="21" cy="7" r="2" fill="currentColor"></circle>
                </svg>
            </span>
            <div>
                <div class="sb-opt4-title">BRENT FORECAST</div>
                <div class="sb-opt4-subtitle">Phân tích & Dự báo Giá dầu Brent</div>
            </div>
        </div>
    </div>

    <div class="sb-spot-bento">
        <div class="sb-spot-top">
            <span>Giá khóa sổ ({spot_date})</span>
            <span class="{spot_pct_cls}">{spot_pct:+.2f}%</span>
        </div>
        <div>
            <span class="sb-spot-val">${spot_price:.2f}</span>
            <span class="sb-spot-unit">/ thùng</span>
        </div>
        <div class="sb-range-track" title="Vị trí giá trong biên độ 52 tuần (252 phiên)">
            <div class="sb-range-fill" style="width: {range_pct:.1f}%;"></div>
        </div>
        <div class="sb-range-labels">
            <span>52W L: ${low_52w:.2f}</span>
            <span>H: ${high_52w:.2f}</span>
        </div>
    </div>

    <div class="sb-dual-grid">
        <div class="sb-horizon-card sb-horizon-t1">
            <div class="sb-h-label">Dự báo t+1 ({focus_m})</div>
            <div class="sb-h-price">${pred_t1:.2f}</div>
            <div class="sb-h-delta {t1_cls}">{diff_t1:+.2f}$ ({t1_status})</div>
        </div>
        <div class="sb-horizon-card sb-horizon-t5">
            <div class="sb-h-label">Dự báo t+5 ({focus_m})</div>
            <div class="sb-h-price">${pred_t5:.2f}</div>
            <div class="sb-h-delta {t5_cls}">{diff_t5:+.2f}$ ({t5_status})</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-group-label">Danh mục phân hệ</div>', unsafe_allow_html=True)

    VIEWS = {
        "Tổng quan & Chuỗi thời gian": render_overview_view,
        "Dự báo Kỳ hạn t+1 & t+5": render_forecast_view,
        "So sánh & Phân tích Sai số": render_evaluation_view,
        "Quản trị & Kiểm toán MLOps": render_admin_view,
    }

    selected_view = st.radio(
        label="Chọn màn hình chức năng",
        options=list(VIEWS.keys()),
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border: none; border-top: 1px solid var(--gen-border); margin: 14px 0;">', unsafe_allow_html=True)

    if st.button("Tra cứu 29 đặc trưng & chỉ số", type="secondary", use_container_width=True):
        open_glossary_modal()

    st.caption("Dữ liệu thị trường: EIA / FRED (01/2015 – 09/2026) · Mô hình: Naive, Ridge, Random Forest, XGBoost.")

inject_genesis_theme("dark")
VIEWS[selected_view]()