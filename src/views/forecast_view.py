import pandas as pd
import streamlit as st
from src.services.data_service import get_clean_data
from src.services.forecast_service import (
    get_available_joblib_models,
    get_all_latest_forecasts,
    generate_october_week_forecast,
)
from src.ui.components import (
    render_section_header,
    render_bento_kpis,
    render_genesis_table,
    render_editorial_note,
)
from src.ui.charts import plot_bento_october_forecast


def render_forecast_view():
    clean_df, has_clean = get_clean_data()
    if not has_clean or clean_df.empty:
        st.error("Chưa tìm thấy dữ liệu lịch sử để thực hiện dự báo.")
        return

    avail_models = get_available_joblib_models()

    render_section_header(
        overline="02 / MULTI-HORIZON PRICE FORECAST",
        title="Dự báo Giá dầu Brent Kỳ hạn Ngắn hạn (t+1) và Trung hạn (t+5)",
        description="Tổng hợp kết quả dự báo xu hướng giá dầu Brent cho 1 phiên và 5 phiên giao dịch kế tiếp từ 4 mô hình học máy dựa trên trạng thái thị trường mới nhất."
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

    fc_df = get_all_latest_forecasts(selected_models)
    oct_df = generate_october_week_forecast(method="direct")

    if fc_df.empty:
        st.warning("Chưa thể tạo dự báo từ các mô hình hiện tại.")
        return

    focus_sub = fc_df[fc_df["model"] == focus_model]
    if focus_sub.empty:
        focus_sub = fc_df[fc_df["model"] == fc_df["model"].iloc[0]]

    last_obs_date = str(fc_df["observation_date"].iloc[0])
    last_price = float(fc_df["last_price"].iloc[0])

    row_t1 = focus_sub[focus_sub["horizon"] == 1].iloc[0]
    row_t5 = focus_sub[focus_sub["horizon"] == 5].iloc[0]

    h1_all = fc_df[fc_df["horizon"] == 1]["predicted_price"]
    h5_all = fc_df[fc_df["horizon"] == 5]["predicted_price"]

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
            "meta": f"Biên độ kỳ hạn t+5: ${h5_all.min():.2f} – ${h5_all.max():.2f}",
            "tone": "indigo",
        },
    ])

    with st.container(border=True):
        st.plotly_chart(
            plot_bento_october_forecast(
                hist_df=clean_df.tail(hist_window),
                oct_df=oct_df,
                selected_models=selected_models,
            ),
            use_container_width=True,
        )
        render_editorial_note(
            "Phương pháp dự báo đa kỳ hạn",
            "Các mô hình sử dụng tổ hợp 29 đặc trưng kỹ thuật tại phiên đóng cửa gần nhất (30/09/2026) để ước lượng trực tiếp mức giá kỳ vọng sau 1 phiên (t+1) và 5 phiên (t+5) giao dịch tiếp theo."
        )

    with st.container(border=True):
        st.markdown("#### Bảng đối chiếu chi tiết kết quả dự báo theo từng mô hình")
        st.caption(f"Mốc quan sát cơ sở: {last_obs_date} · Giá đóng cửa thực tế: ${last_price:.2f}/thùng.")

        table_rows = []
        for m in selected_models:
            m_rows = fc_df[fc_df["model"] == m]
            if len(m_rows) == 2:
                r1 = m_rows[m_rows["horizon"] == 1].iloc[0]
                r5 = m_rows[m_rows["horizon"] == 5].iloc[0]
                table_rows.append({
                    "Mô hình": m,
                    "Ngày quan sát gốc": r1["observation_date"],
                    "Giá gốc ($)": f"${r1['last_price']:.2f}",
                    "Dự báo t+1 ($)": f"${r1['predicted_price']:.2f}",
                    "Biến động t+1": f"{r1['absolute_change']:+.2f} USD ({r1['percent_change']:+.2f}%)",
                    "Dự báo t+5 ($)": f"${r5['predicted_price']:.2f}",
                    "Biến động t+5": f"{r5['absolute_change']:+.2f} USD ({r5['percent_change']:+.2f}%)",
                })
        render_genesis_table(pd.DataFrame(table_rows))