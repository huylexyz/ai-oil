import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from src.core.config import TRAIN_END_DATE, VAL_END_DATE

CHART_PALETTE_DARK = {
    "Actual": "#F8FAFC",
    "XGBoost": "#2DD4BF",
    "Random Forest": "#38BDF8",
    "Ridge": "#FBBF24",
    "Naive": "#FB7185",
}

CHART_PALETTE_LIGHT = {
    "Actual": "#0F172A",
    "XGBoost": "#0D9488",
    "Random Forest": "#0284C7",
    "Ridge": "#D97706",
    "Naive": "#E11D48",
}

MA_PALETTE_DARK = {
    5: "#38BDF8",
    10: "#2DD4BF",
    20: "#FBBF24",
    50: "#FB7185",
}

MA_PALETTE_LIGHT = {
    5: "#0284C7",
    10: "#0D9488",
    20: "#D97706",
    50: "#E11D48",
}

MODEL_DASH_STYLES = {
    "XGBoost": "dashdot",
    "Random Forest": "dash",
    "Ridge": "dot",
    "Naive": "dot",
}


def _get_chart_theme():
    is_dark = st.session_state.get("theme_mode", "dark") == "dark"

    text_main = "#F8FAFC" if is_dark else "#0F172A"
    text_sub = "#94A3B8" if is_dark else "#475569"
    grid_col = "#1E2028" if is_dark else "#E2E8F0"
    baseline_col = "#343844" if is_dark else "#CBD5E1"
    hover_bg = "#181A20" if is_dark else "#FFFFFF"
    hover_border = "#343844" if is_dark else "#CBD5E1"

    model_colors = CHART_PALETTE_DARK if is_dark else CHART_PALETTE_LIGHT
    ma_colors = MA_PALETTE_DARK if is_dark else MA_PALETTE_LIGHT

    layout = dict(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", size=12, color=text_sub),
        title_font=dict(family="General Sans, sans-serif", size=15.5, color=text_main),
        title=dict(x=0.0, y=0.96, xanchor="left", yanchor="top"),
        margin=dict(l=12, r=24, t=76, b=18),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=hover_bg,
            bordercolor=hover_border,
            namelength=-1,
            font=dict(family="JetBrains Mono, monospace", size=12, color=text_main),
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=baseline_col,
            linewidth=1,
            mirror=False,
            title_font=dict(family="DM Sans, sans-serif", size=12, color=text_sub),
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color=text_sub),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor=baseline_col,
            spikethickness=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=grid_col,
            gridwidth=1,
            nticks=5,
            showline=False,
            zeroline=False,
            mirror=False,
            tickprefix="$",
            title_font=dict(family="DM Sans, sans-serif", size=12, color=text_sub),
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color=text_sub),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            title_text="",
            font=dict(family="DM Sans, sans-serif", size=11.5, color=text_main),
        ),
    )
    return layout, is_dark, model_colors, ma_colors, text_main


def plot_bento_historical_price(df: pd.DataFrame, selected_mas: list[int], show_splits: bool = True, **kwargs) -> go.Figure:
    layout, is_dark, model_colors, ma_colors, _ = _get_chart_theme()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["observation_date"],
        y=df["price"],
        mode="lines",
        name="Giá đóng cửa Brent",
        line=dict(color=model_colors["Actual"], width=1.9),
        hoverlabel=dict(namelength=-1),
        hovertemplate="$%{y:.2f}",
    ))

    for ma in selected_mas:
        col = f"ma_{ma}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["observation_date"],
                y=df[col],
                mode="lines",
                name=f"MA {ma} phiên",
                line=dict(width=1.6, dash="dash", color=ma_colors.get(ma, "#0284C7")),
                hoverlabel=dict(namelength=-1),
                hovertemplate="$%{y:.2f}",
            ))

    if show_splits and not df.empty:
        y_max = float(df["price"].max()) * 1.04
        split_line = "#8C92A0" if is_dark else "#64748B"
        split_accent = "#38BDF8" if is_dark else "#0284C7"
        fig.add_vline(x=TRAIN_END_DATE, line_dash="dot", line_color=split_line, line_width=1)
        fig.add_vline(x=VAL_END_DATE, line_dash="dot", line_color=split_accent, line_width=1)
        fig.add_annotation(x="2019-01-01", y=y_max, text="Train (2015–2022)", showarrow=False, font=dict(family="DM Sans", size=11, color=split_line))
        fig.add_annotation(x="2023-12-31", y=y_max, text="Val (2023–2024)", showarrow=False, font=dict(family="DM Sans", size=11, color=split_line))
        fig.add_annotation(x="2025-11-01", y=y_max, text="Test (2025–2026)", showarrow=False, font=dict(family="DM Sans", size=11, color=split_accent))

    fig.update_layout(
        **layout,
        title_text="Diễn biến giá dầu Brent giao ngay & trung bình trượt (USD/thùng)",
        height=420,
    )
    return fig


def plot_bento_october_forecast(hist_df: pd.DataFrame, oct_df: pd.DataFrame, selected_models: list[str], **kwargs) -> go.Figure:
    layout, is_dark, model_colors, _, _ = _get_chart_theme()

    if hist_df.empty or oct_df.empty:
        raise ValueError("Thiếu dữ liệu giá quan sát hoặc dự báo")
    if oct_df["horizon"].tolist() != [1, 2, 3, 4, 5]:
        raise ValueError("Biểu đồ cần đủ dự báo trực tiếp cho phiên t+1 đến t+5")
    missing_models = set(selected_models).difference(oct_df.columns)
    if missing_models:
        raise ValueError(f"Thiếu dự báo của mô hình: {sorted(missing_models)}")

    fig = go.Figure()
    hist_dates = pd.to_datetime(hist_df["observation_date"])
    last_hist_date = hist_dates.iloc[-1]
    last_hist_price = float(hist_df["price"].iloc[-1])
    history_positions = list(range(1 - len(hist_df), 1))
    forecast_positions = oct_df["horizon"].tolist()
    zone_accent = "#38BDF8" if is_dark else "#0284C7"

    fig.add_vrect(
        x0=0,
        x1=5.5,
        fillcolor="rgba(56, 189, 248, 0.06)" if is_dark else "rgba(2, 132, 199, 0.05)",
        layer="below",
        line_width=0,
        annotation_text="Dự báo 5 phiên tiếp theo",
        annotation_position="top left",
        annotation_font=dict(family="DM Sans, sans-serif", size=11.5, color=zone_accent),
    )
    fig.add_vline(x=0, line_dash="dash", line_color=zone_accent, line_width=1.2)

    fig.add_trace(go.Scatter(
        x=history_positions,
        y=hist_df["price"],
        mode="lines",
        name=f"Giá thực tế (Đến {last_hist_date:%d/%m/%Y})",
        line=dict(color=model_colors["Actual"], width=2.2),
        hoverlabel=dict(namelength=-1),
        customdata=hist_dates.dt.strftime("%Y-%m-%d"),
        hovertemplate="%{customdata}<br>$%{y:.2f}<extra></extra>",
    ))

    all_y = hist_df["price"].tolist()
    endpoints = []

    for m in selected_models:
        x_vals = [0] + forecast_positions
        y_vals = [last_hist_price] + oct_df[m].tolist()
        all_y.extend(y_vals)
        m_color = model_colors.get(m, zone_accent)
        end_val = float(oct_df[m].iloc[-1])
        endpoints.append((m, end_val, m_color))

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name=f"Dự báo: {m}",
            line=dict(width=2.1, dash=MODEL_DASH_STYLES.get(m, "dash"), color=m_color),
            marker=dict(size=5, color=m_color),
            hoverlabel=dict(namelength=-1),
            customdata=[f"Ngày quan sát {last_hist_date:%Y-%m-%d}"] + [f"Phiên t+{h}" for h in forecast_positions],
            hovertemplate="%{customdata}<br>$%{y:.2f}<extra></extra>",
        ))

    min_y, max_y = min(all_y), max(all_y)
    span_y = max(max_y - min_y, 4.0)
    pad = max(span_y * 0.22, 1.8)

    # Sắp xếp nhãn giá t+5 có khoảng cách tối thiểu để không đè lên nhau.
    endpoints.sort(key=lambda item: item[1])
    min_gap = span_y * 0.065
    prev_label_y = None
    for m_name, raw_val, m_col in endpoints:
        lbl_y = raw_val if (prev_label_y is None or raw_val - prev_label_y >= min_gap) else (prev_label_y + min_gap)
        prev_label_y = lbl_y
        fig.add_annotation(
            x=5,
            y=lbl_y,
            text=f"<b>${raw_val:.2f}</b>",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(family="JetBrains Mono, monospace", size=11, color=m_col),
        )

    fig.update_layout(
        **layout,
        title_text="Quỹ đạo giá Brent quan sát và dự báo 5 phiên hợp lệ tiếp theo",
        height=420,
    )
    history_ticks = sorted(set(np.linspace(0, len(hist_df) - 1, min(5, len(hist_df)), dtype=int).tolist()))
    tick_values = [history_positions[index] for index in history_ticks] + forecast_positions
    tick_labels = [hist_dates.iloc[index].strftime("%d/%m") for index in history_ticks] + [
        f"t+{h}" for h in forecast_positions
    ]
    fig.update_xaxes(tickmode="array", tickvals=tick_values, ticktext=tick_labels, range=[history_positions[0], 5.6])
    fig.update_yaxes(range=[min_y - pad, max(max_y + pad, (prev_label_y or max_y) + pad * 0.6)])
    return fig


def plot_bento_test_comparison(
    df_sub: pd.DataFrame,
    selected_models: list[str],
    view_mode: str = "focus",
    focus_model: str = None,
    zoom_sessions: int = 60,
    **kwargs,
) -> go.Figure:
    layout, is_dark, model_colors, _, _ = _get_chart_theme()
    if df_sub.empty or not selected_models:
        return go.Figure()

    all_dates = sorted(df_sub["target_date"].unique())
    if zoom_sessions and zoom_sessions < len(all_dates):
        cutoff = all_dates[-zoom_sessions]
        df_plot = df_sub[df_sub["target_date"] >= cutoff].copy()
    else:
        df_plot = df_sub.copy()

    if not focus_model or focus_model not in selected_models:
        focus_model = selected_models[0]

    if view_mode == "facet":
        n_models = len(selected_models)
        cols = 2 if n_models > 1 else 1
        rows = int(np.ceil(n_models / cols))
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.12,
            horizontal_spacing=0.05,
            subplot_titles=[f"Thực tế vs {m}" for m in selected_models],
        )
        for idx, m in enumerate(selected_models):
            r = (idx // cols) + 1
            c = (idx % cols) + 1
            m_df = df_plot[df_plot["model"] == m].sort_values("target_date")
            if not m_df.empty:
                m_color = model_colors.get(m, "#0284C7")
                fig.add_trace(
                    go.Scatter(
                        x=m_df["target_date"],
                        y=m_df["actual"],
                        mode="lines",
                        name="Giá thực tế (Actual)",
                        line=dict(color=model_colors["Actual"], width=1.9),
                        showlegend=(idx == 0),
                        hoverlabel=dict(namelength=-1),
                        hovertemplate="$%{y:.2f}",
                    ),
                    row=r, col=c,
                )
                fig.add_trace(
                    go.Scatter(
                        x=m_df["target_date"],
                        y=m_df["predicted"],
                        mode="lines",
                        name=f"Dự báo: {m}",
                        line=dict(color=m_color, width=2.0, dash="dash"),
                        hoverlabel=dict(namelength=-1),
                        hovertemplate="$%{y:.2f}",
                    ),
                    row=r, col=c,
                )
        fig.update_layout(
            **layout,
            title_text=f"Tách lớp độc lập từng mô hình ({len(df_plot['target_date'].unique())} phiên)",
            height=260 * rows + 80,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(tickprefix="$", nticks=5)
        return fig

    if view_mode == "spread":
        fig = go.Figure()
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color=model_colors["Actual"],
            line_width=1.4,
            annotation_text="Mốc giá thực tế (0 USD)",
            annotation_position="top left",
        )
        for m in selected_models:
            m_df = df_plot[df_plot["model"] == m].sort_values("target_date")
            if not m_df.empty:
                is_focus = (m == focus_model)
                fig.add_trace(
                    go.Scatter(
                        x=m_df["target_date"],
                        y=m_df["error"],
                        mode="lines",
                        name=f"Trọng tâm: {m}" if is_focus else f"{m} (Tham chiếu nền)",
                        line=dict(
                            width=2.3 if is_focus else 1.6,
                            dash="solid" if is_focus else MODEL_DASH_STYLES.get(m, "dot"),
                            color=model_colors.get(m, "#0284C7"),
                        ),
                        opacity=1.0 if is_focus else 0.85,
                        hoverlabel=dict(namelength=-1),
                        hovertemplate="%{y:+.2f} USD",
                    )
                )
        fig.update_layout(
            **layout,
            title_text=f"Độ lệch dự báo so với thực tế (ŷ - y) — Trọng tâm: {focus_model}",
            yaxis_title="Độ lệch ŷ - y (USD)",
            height=420,
        )
        return fig

    fig = go.Figure()
    base_df = df_plot[df_plot["model"] == focus_model].sort_values("target_date")
    if base_df.empty:
        base_df = df_plot[df_plot["model"] == selected_models[0]].sort_values("target_date")

    if not base_df.empty:
        f_color = model_colors.get(focus_model, "#0284C7")
        fig.add_trace(
            go.Scatter(
                x=base_df["target_date"],
                y=base_df["predicted"],
                mode="lines",
                name=f"Trọng tâm: {focus_model}",
                line=dict(width=2.3, dash="dash", color=f_color),
                hoverlabel=dict(namelength=-1),
                hovertemplate="$%{y:.2f}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=base_df["target_date"],
                y=base_df["actual"],
                mode="lines",
                name="Giá thực tế (Actual)",
                line=dict(color=model_colors["Actual"], width=2.4),
                hoverlabel=dict(namelength=-1),
                hovertemplate="$%{y:.2f}",
            )
        )

    for m in selected_models:
        if m == focus_model:
            continue
        m_df = df_plot[df_plot["model"] == m].sort_values("target_date")
        if not m_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=m_df["target_date"],
                    y=m_df["predicted"],
                    mode="lines",
                    name=f"{m} (Tham chiếu nền)",
                    line=dict(
                        width=1.6,
                        dash=MODEL_DASH_STYLES.get(m, "dot"),
                        color=model_colors.get(m, "#0D9488"),
                    ),
                    opacity=0.85,
                    hoverlabel=dict(namelength=-1),
                    hovertemplate="$%{y:.2f}",
                )
            )

    fig.update_layout(
        **layout,
        title_text=f"Đối chiếu giá thực tế & mô hình trọng tâm: {focus_model} ({len(base_df)} phiên)",
        height=420,
    )
    return fig


def plot_bento_error_bars(df_metrics: pd.DataFrame, **kwargs) -> go.Figure:
    layout, is_dark, _, _, text_main = _get_chart_theme()
    text_sub = "#94A3B8" if is_dark else "#475569"
    mae_col = "#38BDF8" if is_dark else "#0284C7"
    rmse_col = "#2DD4BF" if is_dark else "#0D9488"

    melted = df_metrics.melt(
        id_vars="Mô hình",
        value_vars=["MAE (USD)", "RMSE (USD)"],
        var_name="Chỉ số",
        value_name="Sai số (USD)"
    )
    fig = px.bar(
        melted,
        x="Mô hình",
        y="Sai số (USD)",
        color="Chỉ số",
        barmode="group",
        text_auto=".3f",
        color_discrete_map={"MAE (USD)": mae_col, "RMSE (USD)": rmse_col},
    )
    fig.update_traces(
        textfont=dict(family="JetBrains Mono, monospace", size=11, color=text_main),
        textposition="outside",
        hoverlabel=dict(namelength=-1),
    )
    max_y = float(melted["Sai số (USD)"].max()) * 1.22 if not melted.empty else 10.0
    fig.update_layout(
        **layout,
        title_text="Đối chiếu sai số MAE và RMSE (USD/thùng)",
        xaxis_title="Mô hình",
        yaxis_title="Sai số (USD)",
        legend_title_text="",
        height=360,
    )
    fig.update_xaxes(title_font=dict(family="DM Sans, sans-serif", size=12, color=text_sub), tickfont=dict(color=text_main))
    fig.update_yaxes(range=[0, max_y], title_font=dict(family="DM Sans, sans-serif", size=12, color=text_sub), tickfont=dict(color=text_sub))
    return fig


def plot_bento_error_timeline(
    df_sub: pd.DataFrame,
    selected_models: list[str],
    smooth_window: int = 5,
    zoom_sessions: int = 60,
    **kwargs,
) -> go.Figure:
    layout, _, model_colors, _, _ = _get_chart_theme()
    fig = go.Figure()

    all_dates = sorted(df_sub["target_date"].unique())
    if zoom_sessions and zoom_sessions < len(all_dates):
        cutoff = all_dates[-zoom_sessions]
        df_plot = df_sub[df_sub["target_date"] >= cutoff].copy()
    else:
        df_plot = df_sub.copy()

    for m in selected_models:
        m_df = df_plot[df_plot["model"] == m].sort_values("target_date").copy()
        if not m_df.empty:
            y_series = (
                m_df["abs_error"].rolling(smooth_window, min_periods=1).mean()
                if smooth_window > 1
                else m_df["abs_error"]
            )
            fig.add_trace(go.Scatter(
                x=m_df["target_date"],
                y=y_series,
                mode="lines",
                name=m,
                line=dict(width=1.9, color=model_colors.get(m, "#0284C7")),
                hoverlabel=dict(namelength=-1),
                hovertemplate="$%{y:.2f}",
            ))

    title_suffix = f"Trung bình trượt {smooth_window} phiên" if smooth_window > 1 else "Theo từng phiên"
    fig.update_layout(
        **layout,
        title_text=f"Diễn biến sai số tuyệt đối |ŷ - y| ({title_suffix})",
        height=360,
    )
    return fig


def plot_bento_feature_importance(
    imp_df: pd.DataFrame,
    model_name: str,
    horizon: int,
    top_n: int = 15,
    **kwargs,
) -> go.Figure:
    layout, _, model_colors, _, text_main = _get_chart_theme()

    sub = imp_df.head(top_n).sort_values("Tỷ trọng (%)", ascending=True).copy()
    bar_color = model_colors.get(model_name, "#0284C7")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["Tỷ trọng (%)"],
        y=sub["Đặc trưng"],
        orientation="h",
        marker=dict(color=bar_color, line=dict(width=0)),
        text=[f"{v:.2f}%" for v in sub["Tỷ trọng (%)"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color=text_main),
        hoverlabel=dict(namelength=-1),
        hovertemplate="<b>%{y}</b><br>Tỷ trọng đóng góp: %{x:.2f}%<br>Điểm gốc: %{customdata:.5f}<extra></extra>",
        customdata=sub["Điểm gốc"],
    ))

    max_x = float(sub["Tỷ trọng (%)"].max()) * 1.18 if not sub.empty else 100.0
    fig.update_layout(
        **layout,
        title_text=f"Top {len(sub)} đặc trưng ảnh hưởng lớn nhất — {model_name} (Kỳ hạn t+{horizon})",
        xaxis_title="Tỷ trọng đóng góp (%)",
        yaxis_title="",
        height=420,
        showlegend=False,
    )
    fig.update_xaxes(range=[0, max_x], tickprefix="", showgrid=True, gridcolor=layout["yaxis"]["gridcolor"])
    fig.update_yaxes(tickprefix="", showgrid=False)
    return fig
