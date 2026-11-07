import logging
import math
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

sidebar_log = logging.getLogger(__name__)
focus_m = "Random Forest"
try:
    avail_models = get_available_joblib_models()
    if focus_m not in avail_models and avail_models:
        focus_m = avail_models[0]
except Exception as exc:
    sidebar_log.warning("Không kiểm tra được kho model cho sidebar: %s", exc)

spot_date = "không khả dụng"
spot_price_text = spot_pct_text = low_52w_text = high_52w_text = "N/A"
range_width = "0.0%"
spot_pct_cls = "delta-warn"
forecast_cards = {
    1: ("N/A", "Thiếu dữ liệu Brent", "delta-warn"),
    5: ("N/A", "Thiếu dữ liệu Brent", "delta-warn"),
}

try:
    clean_df, has_clean = get_clean_data()
    if not has_clean or not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
        raise ValueError("Thiếu dữ liệu Brent sạch")
    required = {"observation_date", "price"}
    if required.difference(clean_df.columns):
        raise ValueError(f"Dữ liệu Brent thiếu cột {sorted(required.difference(clean_df.columns))}")
    dates = pd.to_datetime(clean_df["observation_date"], errors="raise")
    prices = pd.to_numeric(clean_df["price"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or prices.isna().any():
        raise ValueError("Ngày hoặc giá Brent không hợp lệ")
    if not prices.map(math.isfinite).all() or not prices.gt(0).all():
        raise ValueError("Giá Brent phải hữu hạn và dương")

    clean_df = clean_df.assign(observation_date=dates, price=prices).sort_values("observation_date")
    last_row = clean_df.iloc[-1]
    spot_price = float(last_row["price"])
    observation_date = last_row["observation_date"].strftime("%Y-%m-%d")
    spot_date = last_row["observation_date"].strftime("%d/%m/%Y")
    spot_price_text = f"${spot_price:.2f}"
    if len(clean_df) >= 2:
        previous_price = float(clean_df.iloc[-2]["price"])
        spot_pct = (spot_price - previous_price) / previous_price * 100.0
        spot_pct_text = f"{spot_pct:+.2f}%"
        spot_pct_cls = "delta-pos" if spot_pct >= 0 else "delta-neg"

    recent_252 = clean_df["price"].tail(252)
    low_52w = float(recent_252.min())
    high_52w = float(recent_252.max())
    low_52w_text, high_52w_text = f"${low_52w:.2f}", f"${high_52w:.2f}"
    range_pct = max(4.0, min(96.0, (spot_price - low_52w) / max(high_52w - low_52w, 1e-6) * 100.0))
    range_width = f"{range_pct:.1f}%"
except Exception as exc:
    sidebar_log.warning("Không có dữ liệu Brent hợp lệ cho sidebar: %s", exc)
    reason = "Thiếu dữ liệu Brent" if isinstance(exc, FileNotFoundError) or str(exc).startswith("Thiếu dữ liệu Brent") else "Dữ liệu Brent không hợp lệ"
    forecast_cards[1] = forecast_cards[5] = ("N/A", reason, "delta-warn")
else:
    for horizon in (1, 5):
        try:
            result = get_latest_official_forecast(focus_m, horizon)
            if result["model"] != focus_m or result["horizon"] != horizon:
                raise ValueError("Backend trả sai mô hình hoặc horizon")
            if result["observation_date"] != observation_date or not math.isclose(
                float(result["last_price"]), spot_price, rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError("Ngày hoặc giá quan sát của dự báo không khớp dữ liệu Brent")
            predicted = float(result["predicted_price"])
            change = float(result["absolute_change"])
            if not math.isfinite(predicted) or predicted <= 0 or not math.isfinite(change):
                raise ValueError("Giá dự báo không hợp lệ")
            if not math.isclose(change, predicted - spot_price, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Mức thay đổi dự báo không khớp giá Brent")
            forecast_cards[horizon] = (
                f"${predicted:.2f}",
                f"{change:+.2f}$ ({'Tăng nhẹ' if change >= 0 else 'Điều chỉnh'})",
                "delta-pos" if change >= 0 else "delta-warn",
            )
        except Exception as exc:
            sidebar_log.warning("Dự báo sidebar %s t+%s không khả dụng: %s", focus_m, horizon, exc)
            reason = "Thiếu model/dữ liệu" if isinstance(exc, FileNotFoundError) else "Lỗi dự báo chính thức"
            forecast_cards[horizon] = ("N/A", reason, "delta-warn")

pred_t1_text, t1_delta_text, t1_cls = forecast_cards[1]
pred_t5_text, t5_delta_text, t5_cls = forecast_cards[5]

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
            <span class="{spot_pct_cls}">{spot_pct_text}</span>
        </div>
        <div>
            <span class="sb-spot-val">{spot_price_text}</span>
            <span class="sb-spot-unit">/ thùng</span>
        </div>
        <div class="sb-range-track" title="Vị trí giá trong biên độ 52 tuần (252 phiên)">
            <div class="sb-range-fill" style="width: {range_width};"></div>
        </div>
        <div class="sb-range-labels">
            <span>52W L: {low_52w_text}</span>
            <span>H: {high_52w_text}</span>
        </div>
    </div>

    <div class="sb-dual-grid">
        <div class="sb-horizon-card sb-horizon-t1">
            <div class="sb-h-label">Dự báo t+1 ({focus_m})</div>
            <div class="sb-h-price">{pred_t1_text}</div>
            <div class="sb-h-delta {t1_cls}">{t1_delta_text}</div>
        </div>
        <div class="sb-horizon-card sb-horizon-t5">
            <div class="sb-h-label">Dự báo t+5 ({focus_m})</div>
            <div class="sb-h-price">{pred_t5_text}</div>
            <div class="sb-h-delta {t5_cls}">{t5_delta_text}</div>
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
