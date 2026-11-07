import inspect
import importlib
import pandas as pd
import streamlit as st
from src.core.config import MODELS_LIST, OFFICIAL_HORIZONS
from src.services.data_service import (
    get_all_predictions,
    get_rf_tuning_results,
    get_feature_importance_data,
)
from src.services.forecast_service import compute_evaluation_metrics
from src.ui.components import (
    render_section_header,
    render_bento_kpis,
    render_genesis_table,
)
import src.ui.charts as charts_module
from src.ui.modals import open_glossary_modal


def _format_metrics_for_display(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Format official metrics at the view boundary while retaining raw values."""
    rows = []
    for rank, row in enumerate(raw_metrics.itertuples(index=False), start=1):
        rows.append({
            "Hạng": rank,
            "Mô hình": row.model,
            "Số phiên": int(row.n_samples),
            "MAE (USD)": float(row.mae),
            "RMSE (USD)": float(row.rmse),
            "Directional Accuracy (%)": (
                "N/A" if pd.isna(row.directional_accuracy)
                else f"{row.directional_accuracy:.2f}%"
            ),
        })
    return pd.DataFrame(rows)


def _call_test_comparison_chart(df_sub, selected_models, view_mode, focus_model, zoom_sessions):
    func = getattr(charts_module, "plot_bento_test_comparison")
    params = inspect.signature(func).parameters
    if "view_mode" not in params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        importlib.reload(charts_module)
        func = getattr(charts_module, "plot_bento_test_comparison")
        params = inspect.signature(func).parameters

    if "view_mode" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(
            df_sub=df_sub,
            selected_models=selected_models,
            view_mode=view_mode,
            focus_model=focus_model,
            zoom_sessions=zoom_sessions,
        )
    return func(df_sub, selected_models)


def _call_error_timeline_chart(df_sub, selected_models, smooth_window, zoom_sessions):
    func = getattr(charts_module, "plot_bento_error_timeline")
    params = inspect.signature(func).parameters
    if "smooth_window" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(
            df_sub=df_sub,
            selected_models=selected_models,
            smooth_window=smooth_window,
            zoom_sessions=zoom_sessions,
        )
    return func(df_sub, selected_models)


def render_evaluation_view():
    df_pred, has_pred, _ = get_all_predictions()
    if not has_pred or df_pred.empty:
        st.error("Chưa tìm thấy dữ liệu kiểm định mô hình trên tập Test.")
        return

    head_col, btn_col = st.columns([3.3, 1])
    with head_col:
        render_section_header(
            overline="03 / MODEL BENCHMARKING, ERROR & FEATURE IMPORTANCE",
            title="So sánh Hiệu năng, Độ sai sót & Mức độ Quan trọng Đặc trưng",
            description="Đánh giá thực nghiệm trên 442 phiên kiểm tra độc lập (01/2025 – 09/2026) theo bộ ba chỉ số MAE, RMSE, tỷ lệ dự báo đúng xu hướng (Directional Accuracy) và phân tích trọng số của 29 đặc trưng kỹ thuật."
        )
    with btn_col:
        st.write("")
        if st.button("Giải thích công thức chỉ số", type="secondary", use_container_width=True):
            open_glossary_modal()

    with st.container(border=True):
        f1, f2, f3 = st.columns([1.1, 1.3, 1.6])
        horizon = f1.selectbox("Kỳ hạn dự báo cần đánh giá", OFFICIAL_HORIZONS, format_func=lambda x: f"Kỳ hạn t+{x} ({x} phiên)")

        df_h = df_pred[(df_pred["horizon"] == horizon) & (df_pred["split"] == "test")].copy()
        avail_models = [m for m in MODELS_LIST if m in df_h["model"].unique()]

        time_preset = f2.selectbox(
            "Khoảng thời gian phân tích",
            [
                "Toàn bộ tập Kiểm tra (2025 – 09/2026)",
                "Năm 2025",
                "Năm 2026",
                "6 tháng cuối",
                "3 tháng cuối",
                "Tùy chỉnh ngày",
            ]
        )
        selected_models = f3.multiselect("Mô hình đối chiếu", options=avail_models, default=avail_models)

        if df_h.empty:
            st.warning(f"Chưa có dữ liệu Test chính thức cho kỳ hạn t+{horizon}.")
            return
        if not selected_models:
            st.warning("Vui lòng chọn ít nhất một mô hình để đối chiếu hiệu năng.")
            return

        min_td = df_h["target_date"].min().date()
        max_td = df_h["target_date"].max().date()

        if time_preset == "Năm 2025":
            start_td, end_td = max(min_td, pd.Timestamp("2025-01-01").date()), min(max_td, pd.Timestamp("2025-12-31").date())
        elif time_preset == "Năm 2026":
            start_td, end_td = max(min_td, pd.Timestamp("2026-01-01").date()), max_td
        elif time_preset == "6 tháng cuối":
            start_td, end_td = max(min_td, (pd.Timestamp(max_td) - pd.DateOffset(months=6)).date()), max_td
        elif time_preset == "3 tháng cuối":
            start_td, end_td = max(min_td, (pd.Timestamp(max_td) - pd.DateOffset(months=3)).date()), max_td
        elif time_preset == "Tùy chỉnh ngày":
            c_range = st.date_input("Khoảng ngày mục tiêu", value=(min_td, max_td), min_value=min_td, max_value=max_td)
            start_td, end_td = (c_range[0], c_range[1]) if len(c_range) == 2 else (min_td, max_td)
        else:
            start_td, end_td = min_td, max_td

    df_sub = df_h[
        (df_h["model"].isin(selected_models))
        & (df_h["target_date"].dt.date >= start_td)
        & (df_h["target_date"].dt.date <= end_td)
    ].copy()

    if df_sub.empty:
        st.warning("Không có bản ghi nào trong mốc thời gian đã chọn.")
        return

    raw_metrics = compute_evaluation_metrics(df_sub, selected_models)
    if raw_metrics.empty:
        st.warning("Không có chỉ số đánh giá cho dữ liệu Test đã chọn.")
        return
    display_metrics = _format_metrics_for_display(raw_metrics)

    best_rmse = raw_metrics.iloc[0]
    best_mae = raw_metrics.loc[raw_metrics["mae"].idxmin()]
    da_candidates = raw_metrics.loc[raw_metrics["directional_accuracy"].notna()]

    if da_candidates.empty:
        best_da_model, best_da_val = "N/A", "N/A"
    else:
        best_da_row = da_candidates.loc[da_candidates["directional_accuracy"].idxmax()]
        best_da_model = str(best_da_row["model"])
        best_da_val = f"{best_da_row['directional_accuracy']:.2f}%"

    render_bento_kpis([
        {
            "overline": f"Mô hình tối ưu nhất (RMSE t+{horizon})",
            "badge": "Hạng 1",
            "value": str(best_rmse["model"]),
            "meta": f"RMSE: ${best_rmse['rmse']:.4f} / thùng",
            "tone": "success",
        },
        {
            "overline": "Sai số tuyệt đối thấp nhất (MAE)",
            "badge": str(best_mae["model"]),
            "value": f"${best_mae['mae']:.4f}",
            "meta": "Sai lệch trung bình mỗi phiên",
            "tone": "neutral",
        },
        {
            "overline": "Dự báo đúng xu hướng cao nhất (DA)",
            "badge": best_da_model,
            "value": best_da_val,
            "meta": "Tỷ lệ dự báo đúng chiều tăng / giảm",
            "tone": "success",
        },
        {
            "overline": "Phạm vi kiểm định độc lập",
            "badge": f"Kỳ hạn t+{horizon}",
            "value": f"{best_rmse['n_samples']} phiên",
            "meta": f"{start_td.strftime('%Y-%m-%d')} → {end_td.strftime('%Y-%m-%d')}",
            "tone": "neutral",
        },
    ])

    c_table, c_bar = st.columns([1.25, 1.05])
    with c_table:
        with st.container(border=True):
            st.markdown(f"#### Bảng xếp hạng hiệu năng các mô hình (Kỳ hạn t+{horizon})")
            st.caption(f"Đánh giá từ ngày {start_td} đến {end_td} ({best_rmse['n_samples']} phiên giao dịch độc lập).")
            render_genesis_table(display_metrics)
    with c_bar:
        with st.container(border=True):
            st.plotly_chart(charts_module.plot_bento_error_bars(display_metrics), use_container_width=True)

    with st.container(border=True):
        total_sessions = max(1, int(best_rmse["n_samples"]))
        min_zoom = min(20, total_sessions)
        default_zoom = min(60, total_sessions)

        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.45, 1.05, 1.15, 0.95])
        view_mode_label = ctrl1.selectbox(
            "Chế độ hiển thị biểu đồ",
            options=["focus", "facet", "spread"],
            format_func=lambda x: {
                "focus": "Tập trung 1 mô hình trọng tâm (Khuyên dùng)",
                "facet": "Chia lưới 2x2 độc lập từng mô hình",
                "spread": "Độ lệch so với giá thực tế (ŷ - y)",
            }[x],
        )
        focus_model_choice = ctrl2.selectbox(
            "Mô hình làm nổi bật",
            options=selected_models,
            index=selected_models.index(str(best_rmse["model"])) if str(best_rmse["model"]) in selected_models else 0,
        )
        if total_sessions > min_zoom:
            zoom_sessions = ctrl3.slider(
                "Số phiên hiển thị (Phóng đại)",
                min_value=min_zoom,
                max_value=total_sessions,
                value=default_zoom,
                step=10 if total_sessions >= 40 else 1,
            )
        else:
            zoom_sessions = total_sessions

        smooth_window = ctrl4.selectbox(
            "Làm mượt biểu đồ sai số",
            options=[5, 10, 1],
            format_func=lambda x: f"TB trượt {x} phiên" if x > 1 else "Gốc (1 phiên)",
        )

        st.plotly_chart(
            _call_test_comparison_chart(
                df_sub=df_sub,
                selected_models=selected_models,
                view_mode=view_mode_label,
                focus_model=focus_model_choice,
                zoom_sessions=zoom_sessions,
            ),
            use_container_width=True,
        )

    e_left, e_right = st.columns([1.3, 1])
    with e_left:
        with st.container(border=True):
            st.plotly_chart(
                _call_error_timeline_chart(
                    df_sub=df_sub,
                    selected_models=selected_models,
                    smooth_window=smooth_window,
                    zoom_sessions=zoom_sessions,
                ),
                use_container_width=True,
            )
    with e_right:
        with st.container(border=True):
            st.markdown("#### Các phiên biến động gây sai số lớn nhất")
            st.caption("Top 6 phiên có độ lệch tuyệt đối |ŷ - y| cao nhất trong giai đoạn đang lọc.")
            top_err = df_sub.sort_values("abs_error", ascending=False)[
                ["target_date", "model", "actual", "predicted", "abs_error"]
            ].head(6).copy()
            top_err["target_date"] = top_err["target_date"].dt.strftime("%Y-%m-%d")
            for c in ["actual", "predicted", "abs_error"]:
                top_err[c] = top_err[c].map(lambda x: f"${x:.2f}")
            top_err.columns = ["Ngày mục tiêu", "Mô hình", "Thực tế ($)", "Dự báo ($)", "|ŷ - y| ($)"]
            render_genesis_table(top_err)

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"#### Phân tích mức độ quan trọng của 29 đặc trưng (Kỳ hạn t+{horizon})")
        st.caption(
            "Đánh giá tỷ trọng đóng góp của các biến đầu vào đối với mô hình học máy. "
            "Lưu ý: Trọng số đặc trưng phản ánh mức độ sử dụng biến khi phân tách cây/hồi quy, không biểu thị chiều tác động nhân quả."
        )

        fi_col1, fi_col2 = st.columns([1.2, 1.2])
        imp_candidates = ["XGBoost", "Random Forest", "Ridge"]
        default_imp_idx = 0
        for idx, m_cand in enumerate(imp_candidates):
            test_imp, _ = get_feature_importance_data(m_cand, horizon)
            if not test_imp.empty:
                default_imp_idx = idx
                break

        imp_model = fi_col1.selectbox(
            "Chọn mô hình phân tích đặc trưng",
            options=imp_candidates,
            index=default_imp_idx,
        )
        top_n_feats = fi_col2.slider(
            "Số lượng đặc trưng hiển thị trên biểu đồ",
            min_value=5,
            max_value=29,
            value=15,
            step=1,
        )

        imp_df, imp_source = get_feature_importance_data(imp_model, horizon)
        if imp_df.empty:
            st.info(f"Chưa có dữ liệu trọng số đặc trưng của mô hình {imp_model} (kỳ hạn t+{horizon}).")
        else:
            chart_col, table_col = st.columns([1.35, 1])
            with chart_col:
                st.plotly_chart(
                    charts_module.plot_bento_feature_importance(imp_df, imp_model, horizon, top_n=top_n_feats),
                    use_container_width=True
                )
            with table_col:
                st.markdown(f"**Bảng xếp hạng đặc trưng ({imp_source})**")
                top3_names = ", ".join(imp_df["Đặc trưng"].head(3).tolist())
                st.caption(f"3 đặc trưng chi phối mạnh nhất: **{top3_names}** (chiếm **{imp_df['Tỷ trọng (%)'].head(3).sum():.2f}%** tổng trọng số).")
                render_genesis_table(imp_df, max_height=340)

    rf_tuning = get_rf_tuning_results(horizon)
    if not rf_tuning.empty:
        with st.expander(f"Xem bảng kết quả tinh chỉnh siêu tham số Random Forest trên tập Validation (Kỳ hạn t+{horizon})"):
            render_genesis_table(rf_tuning.sort_values("validation_RMSE"))
