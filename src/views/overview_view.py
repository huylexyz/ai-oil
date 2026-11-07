import math

import pandas as pd
import streamlit as st
from src.core.config import MAS, TRAIN_END_DATE, VAL_END_DATE
from src.services.data_service import get_clean_data, get_removed_rows, get_features_data
from src.ui.components import (
    render_section_header,
    render_bento_kpis,
    render_genesis_table,
    render_editorial_note,
)
from src.ui.charts import plot_bento_historical_price
from src.ui.modals import open_glossary_modal


def render_overview_view():
    try:
        clean_df, has_clean = get_clean_data()
        if not has_clean or not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
            raise ValueError("Chưa tìm thấy bộ dữ liệu lịch sử Brent đã làm sạch.")
        required = {"observation_date", "price"}
        missing = required.difference(clean_df.columns)
        if missing:
            raise ValueError(f"Dữ liệu Brent thiếu cột bắt buộc: {sorted(missing)}")
        dates = pd.to_datetime(clean_df["observation_date"], errors="raise")
        prices = pd.to_numeric(clean_df["price"], errors="raise")
        if dates.isna().any() or dates.duplicated().any() or prices.isna().any():
            raise ValueError("Dữ liệu Brent có ngày hoặc giá bị thiếu hay trùng lặp.")
        if not prices.map(math.isfinite).all() or not prices.gt(0).all():
            raise ValueError("Giá Brent phải là số hữu hạn và dương.")
        if len(clean_df) < 2:
            raise ValueError("Cần ít nhất hai phiên Brent hợp lệ để tính thay đổi giá.")
        clean_df = clean_df.assign(observation_date=dates, price=prices).sort_values("observation_date").reset_index(drop=True)
    except Exception as exc:
        st.error(f"Không thể hiển thị dữ liệu Brent lịch sử: {exc}")
        return

    try:
        feat_df, has_feat = get_features_data()
        if not has_feat or not isinstance(feat_df, pd.DataFrame) or feat_df.empty:
            raise ValueError("chưa có dữ liệu đặc trưng")
        if not {"observation_date", "price"}.issubset(feat_df.columns):
            raise ValueError("dữ liệu đặc trưng thiếu ngày quan sát hoặc giá")
        feature_dates = pd.to_datetime(feat_df["observation_date"], errors="raise")
        feature_prices = pd.to_numeric(feat_df["price"], errors="raise")
        if feature_dates.isna().any() or feature_prices.isna().any() or not feature_prices.map(math.isfinite).all():
            raise ValueError("dữ liệu đặc trưng có ngày hoặc giá không hợp lệ")
        feat_df = feat_df.assign(observation_date=feature_dates, price=feature_prices)
    except Exception as exc:
        st.warning(f"Không thể hiển thị bộ đặc trưng; biểu đồ dùng giá Brent sạch: {exc}")
        feat_df, has_feat = pd.DataFrame(), False

    try:
        rem_df_res, has_rem = get_removed_rows()
        rem_df = rem_df_res if has_rem and isinstance(rem_df_res, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:
        st.warning(f"Không thể đọc nhật ký các phiên đã loại bỏ: {exc}")
        rem_df, has_rem = pd.DataFrame(), False

    df_plot_base = feat_df if (has_feat and not feat_df.empty) else clean_df

    head_col, btn_col = st.columns([3.3, 1])
    with head_col:
        render_section_header(
            overline="01 / HISTORICAL MARKET DATA",
            title="Tổng quan Dữ liệu Giá dầu Brent (2015 – 09/2026)",
            description=f"Chuỗi dữ liệu thị trường chuẩn hóa từ nguồn EIA/FRED gồm {len(clean_df):,} phiên giao dịch hợp lệ, kết hợp 29 đặc trưng kỹ thuật và phân chia ranh giới thời gian nghiêm ngặt."
        )
    with btn_col:
        st.write("")
        if st.button("Mở Từ điển 29 Đặc trưng", type="secondary", use_container_width=True):
            open_glossary_modal()

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1.5, 1.1])
        period_choice = c1.selectbox(
            "Phân vùng thời gian quan sát",
            options=[
                "Toàn bộ lịch sử (01/2015 – 09/2026)",
                "Tập Huấn luyện — Train (≤ 2022)",
                "Tập Tinh chỉnh — Validation (2023 – 2024)",
                "Tập Kiểm tra — Test (2025 – 09/2026)",
            ],
        )
        selected_mas = c2.multiselect(
            "Đường trung bình trượt (Moving Averages)",
            options=MAS,
            default=[20, 50],
            format_func=lambda x: f"MA {x} phiên",
        )
        c3.write("")
        show_splits = c3.checkbox("Hiển thị ranh giới Train / Val / Test", value=True)

    df_view = df_plot_base.copy()
    if "Train" in period_choice:
        df_view = df_view[df_view["observation_date"] <= TRAIN_END_DATE]
    elif "Validation" in period_choice:
        df_view = df_view[(df_view["observation_date"] > TRAIN_END_DATE) & (df_view["observation_date"] <= VAL_END_DATE)]
    elif "Test" in period_choice:
        df_view = df_view[df_view["observation_date"] > VAL_END_DATE]

    last_row = clean_df.iloc[-1]
    prev_row = clean_df.iloc[-2]
    last_p = float(last_row["price"])
    diff_p = last_p - float(prev_row["price"])
    pct_p = (diff_p / float(prev_row["price"])) * 100.0

    min_row = clean_df.loc[clean_df["price"].idxmin()]
    max_row = clean_df.loc[clean_df["price"].idxmax()]
    n_test = int((feat_df["split_1"] == "test").sum()) if has_feat and "split_1" in feat_df.columns else None
    n_rem = len(rem_df) if has_rem else None
    removed_text = str(n_rem) if n_rem is not None else "N/A"

    render_bento_kpis([
        {
            "overline": "Giá phiên cuối kỳ",
            "badge": f"{diff_p:+.2f} USD ({pct_p:+.2f}%)",
            "value": f"${last_p:.2f}",
            "meta": f"Ngày quan sát: {pd.to_datetime(last_row['observation_date']).strftime('%Y-%m-%d')}",
            "tone": "success" if diff_p >= 0 else "error",
        },
        {
            "overline": "Biên độ trong kỳ (MIN – MAX)",
            "badge": f"TB: ${clean_df['price'].mean():.2f}",
            "value": f"${clean_df['price'].min():.2f} – ${clean_df['price'].max():.2f}",
            "meta": f"Thấp nhất ({pd.to_datetime(min_row['observation_date']).strftime('%Y-%m-%d')}) · Cao nhất ({pd.to_datetime(max_row['observation_date']).strftime('%Y-%m-%d')})",
            "tone": "neutral",
        },
        {
            "overline": "Quy mô mẫu đặc trưng",
            "badge": "29 Đặc trưng" if has_feat else "Thiếu đặc trưng",
            "value": f"{len(feat_df):,} phiên" if has_feat else "N/A",
            "meta": f"Từ {len(clean_df):,} phiên hợp lệ (đã lọc {removed_text} phiên thiếu giá)",
            "tone": "indigo" if has_feat else "warning",
        },
        {
            "overline": "Tập kiểm tra độc lập",
            "badge": "Chống rò rỉ",
            "value": f"{n_test} phiên" if n_test is not None else "N/A",
            "meta": "Giai đoạn 01/2025 – 09/2026 (Phân chia theo ngày mục tiêu)",
            "tone": "success" if n_test is not None else "warning",
        },
    ])

    with st.container(border=True):
        st.plotly_chart(
            plot_bento_historical_price(df_view, selected_mas, show_splits=show_splits),
            use_container_width=True,
        )
        render_editorial_note(
            "Quy chuẩn xử lý dữ liệu & đặc trưng",
            f"Từ {len(clean_df) + n_rem:,} bản ghi gốc của EIA/FRED, hệ thống loại bỏ {n_rem} phiên thiếu giá để giữ lại {len(clean_df):,} phiên hợp lệ ({clean_df['observation_date'].min():%d/%m/%Y} – {clean_df['observation_date'].max():%d/%m/%Y}); bộ đặc trưng có {len(feat_df):,} phiên."
            if n_rem is not None and has_feat else
            f"Có {len(clean_df):,} phiên Brent hợp lệ ({clean_df['observation_date'].min():%d/%m/%Y} – {clean_df['observation_date'].max():%d/%m/%Y}); nhật ký phiên thiếu giá hoặc bộ đặc trưng hiện không đầy đủ."
        )

    b1, b2 = st.columns([1.25, 1])
    with b1:
        with st.container(border=True):
            st.markdown("#### Bảng tra cứu lịch sử giá & chỉ báo kỹ thuật (15 phiên gần nhất)")
            st.caption("Hiển thị giá đóng cửa cùng các chỉ báo độ trễ, trung bình động và biến động đặc trưng.")
            cols_show = ["observation_date", "price", "lag_1", "ma_5", "ma_20", "ma_50", "ret_1", "vol_20"]
            avail_c = [c for c in cols_show if c in df_view.columns]
            sub_t = df_view[avail_c].tail(15).sort_values("observation_date", ascending=False).copy()
            sub_t["observation_date"] = pd.to_datetime(sub_t["observation_date"]).dt.strftime("%Y-%m-%d")
            render_genesis_table(sub_t, max_height=320)
    with b2:
        with st.container(border=True):
            st.markdown("#### Nhật ký các ngày nghỉ lễ thị trường đã loại bỏ")
            st.caption(f"Tổng cộng {removed_text} phiên thiếu giá được ghi nhận và lọc khỏi chuỗi huấn luyện.")
            if not rem_df.empty:
                render_genesis_table(rem_df.head(15), max_height=320)
