import streamlit as st

def inject_genesis_theme(mode: str = "dark"):
    """ Tùy chỉnh cho giao diện Swiss Institutional Slate (Dark | Light)"""
    is_dark = (mode == "dark")
    scheme_name = "dark" if is_dark else "light"

    bg_color = "#0A0B0D" if is_dark else "#F8FAFC"
    sidebar_bg = "#101115" if is_dark else "#FFFFFF"
    surface_color = "#121317" if is_dark else "#FFFFFF"
    input_bg = "#181A20" if is_dark else "#F1F5F9"
    surface_hover = "#21242C" if is_dark else "#E2E8F0"

    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#B0B5C0" if is_dark else "#475569"
    text_faint = "#8C92A0" if is_dark else "#64748B"

    border_color = "#24262D" if is_dark else "#E2E8F0"
    input_border = "#343844" if is_dark else "#CBD5E1"
    rule_color = "#1B1D24" if is_dark else "#F1F5F9"

    accent_color = "#38BDF8" if is_dark else "#0284C7"
    accent_bg = "rgba(56, 189, 248, 0.15)" if is_dark else "#E0F2FE"
    accent_fg = "#38BDF8" if is_dark else "#0369A1"
    accent_bd = "rgba(56, 189, 248, 0.35)" if is_dark else "#BAE6FD"

    t1_accent = "#38BDF8" if is_dark else "#0284C7"
    t5_accent = "#2DD4BF" if is_dark else "#0D9488"

    pos_color = "#34D399" if is_dark else "#15803D"
    pos_bg = "rgba(52, 211, 153, 0.14)" if is_dark else "#DCFCE7"
    pos_bd = "rgba(52, 211, 153, 0.35)" if is_dark else "#BBF7D0"

    neg_color = "#F87171" if is_dark else "#B91C1C"
    neg_bg = "rgba(248, 113, 113, 0.14)" if is_dark else "#FEE2E2"
    neg_bd = "rgba(248, 113, 113, 0.35)" if is_dark else "#FECACA"

    warn_color = "#FBBF24" if is_dark else "#B45309"
    warn_bg = "rgba(251, 191, 36, 0.14)" if is_dark else "#FEF3C7"
    warn_bd = "rgba(251, 191, 36, 0.35)" if is_dark else "#FDE68A"

    primary_btn_bg = "#38BDF8" if is_dark else "#0F172A"
    primary_btn_fg = "#0A0B0D" if is_dark else "#FFFFFF"

    st.markdown(f"""
    <style>
        @import url('https://api.fontshare.com/v2/css?f[]=general-sans@500,600,700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root, html, body, .stApp {{
            color-scheme: {scheme_name} !important;
            --primary-color: {accent_color} !important;
            --background-color: {bg_color} !important;
            --secondary-background-color: {surface_color} !important;
            --text-color: {text_primary} !important;

            --gen-primary: {accent_color};
            --gen-primary-bg: {accent_bg};
            --gen-primary-fg: {accent_fg};
            --gen-primary-bd: {accent_bd};
            --gen-bg: {bg_color};
            --gen-sidebar: {sidebar_bg};
            --gen-surface: {surface_color};
            --gen-surface-alt: {input_bg};
            --gen-surface-hover: {surface_hover};
            --gen-text-primary: {text_primary};
            --gen-text-secondary: {text_secondary};
            --gen-text-faint: {text_faint};
            --gen-border: {border_color};
            --gen-control-border: {input_border};
            --gen-rule: {rule_color};
            --gen-success: {pos_color};
            --gen-success-bg: {pos_bg};
            --gen-success-bd: {pos_bd};
            --gen-warning: {warn_color};
            --gen-warning-bg: {warn_bg};
            --gen-warning-bd: {warn_bd};
            --gen-error: {neg_color};
            --gen-error-bg: {neg_bg};
            --gen-error-bd: {neg_bd};
            --t1-accent: {t1_accent};
            --t5-accent: {t5_accent};
        }}

        /* 1. Nền trang chính & Xóa thanh trang trí mặc định */
        .stApp, [data-testid="stAppViewContainer"], .main {{
            background-color: var(--gen-bg) !important;
            background-image: none !important;
            color: var(--gen-text-primary) !important;
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        [data-testid="stHeader"] {{
            background-color: transparent !important;
            height: 2.2rem !important;
        }}
        [data-testid="stToolbar"], [data-testid="stHeader"] * {{
            color: var(--gen-text-secondary) !important;
        }}
        [data-testid="stDecoration"] {{
            display: none !important;
        }}

        .main .block-container {{
            max-width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            padding: 1rem 2rem 3.5rem 1.75rem !important;
        }}

        /* 2. Thanh Sidebar chuẩn Swiss Institutional Slate */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {{
            background-color: var(--gen-sidebar) !important;
            border-right: 1px solid var(--gen-border) !important;
            min-width: 282px !important;
        }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {{
            color: var(--gen-text-primary);
        }}
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {{
            color: var(--gen-text-faint) !important;
        }}

        [data-testid="stSidebarHeader"] {{
            height: 1.5rem !important;
            min-height: 1.5rem !important;
            padding: 0.2rem 0.5rem 0 0.5rem !important;
            margin-bottom: -1.1rem !important;
            background: transparent !important;
        }}
        [data-testid="stSidebarUserContent"] {{
            padding-top: 0rem !important;
        }}
        [data-testid="stSidebarHeader"] button,
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button {{
            color: var(--gen-text-secondary) !important;
            background: transparent !important;
        }}
        [data-testid="stSidebarHeader"] svg,
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="collapsedControl"] svg {{
            fill: var(--gen-text-secondary) !important;
            color: var(--gen-text-secondary) !important;
        }}

        /* Cụm thương hiệu đầu Sidebar */
        .sb-opt4-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }}
        .sb-opt4-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .sb-opt4-mark {{
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background: var(--gen-primary-bg);
            border: 1px solid var(--gen-primary-bd);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .sb-opt4-title {{
            font-family: 'General Sans', sans-serif !important;
            font-weight: 700;
            font-size: 16.5px;
            letter-spacing: -0.025em;
            line-height: 1.1;
            color: var(--gen-text-primary) !important;
        }}
        .sb-opt4-subtitle {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 10.5px;
            color: var(--gen-text-faint) !important;
            margin-top: 2px;
        }}

        /* Thẻ Mốc khóa sổ & Lưới 2 cột Dự báo trên Sidebar */
        .sb-spot-bento {{
            background: var(--gen-surface);
            border: 1px solid var(--gen-border);
            border-radius: 8px;
            padding: 11px 12px;
            margin-bottom: 8px;
        }}
        .sb-spot-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11.5px;
            font-weight: 500;
            color: var(--gen-text-secondary) !important;
            margin-bottom: 4px;
        }}
        .sb-spot-val {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 19px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--gen-text-primary) !important;
            letter-spacing: -0.02em;
        }}
        .sb-spot-unit {{
            font-family: 'DM Sans', sans-serif !important;
            font-size: 12px;
            font-weight: 400;
            color: var(--gen-text-faint) !important;
            margin-left: 4px;
        }}
        .sb-range-track {{
            width: 100%;
            height: 4px;
            background: var(--gen-surface-alt);
            border-radius: 2px;
            position: relative;
            margin: 8px 0 4px 0;
        }}
        .sb-range-fill {{
            height: 100%;
            background: var(--gen-primary);
            border-radius: 2px;
        }}
        .sb-range-labels {{
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 10px;
            color: var(--gen-text-faint) !important;
        }}

        .sb-dual-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .sb-horizon-card {{
            background: var(--gen-surface);
            border: 1px solid var(--gen-border);
            border-radius: 8px;
            padding: 9px 10px;
        }}
        .sb-horizon-t1 {{
            border-left: 3px solid var(--t1-accent) !important;
        }}
        .sb-horizon-t5 {{
            border-left: 3px solid var(--t5-accent) !important;
        }}
        .sb-h-label {{
            font-size: 11px;
            font-weight: 500;
            color: var(--gen-text-secondary) !important;
            margin-bottom: 3px;
        }}
        .sb-h-price {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 15px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--gen-text-primary) !important;
            margin-bottom: 2px;
        }}
        .sb-h-delta {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 10.5px;
            font-weight: 500;
            font-variant-numeric: tabular-nums;
        }}

        /* 3. XÓA TRIỆT ĐỂ CHẤM TRÒN ĐEN CỦA RADIO TRONG SIDEBAR & TẠO NÚT GẠT DARK | LIGHT */
        .sb-group-label {{
            font-size: 11.5px;
            font-weight: 600;
            color: var(--gen-text-faint) !important;
            margin: 6px 0 6px 2px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
        section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        /* Menu dọc: Danh mục phân hệ */
        section[data-testid="stSidebar"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) {{
            display: flex !important;
            flex-direction: column !important;
            gap: 4px !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-left: 3px solid transparent !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            margin: 0 !important;
            width: 100% !important;
            transition: background-color 120ms ease !important;
            cursor: pointer !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background-color: var(--gen-surface-alt) !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {{
            background-color: var(--gen-primary-bg) !important;
            border-color: var(--gen-primary-bd) !important;
            border-left: 3px solid var(--gen-primary) !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{
            font-size: 13.5px !important;
            color: var(--gen-text-secondary) !important;
            margin: 0 !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p {{
            color: var(--gen-primary-fg) !important;
            font-weight: 600 !important;
        }}

        /* Thanh gạt ngang Segmented Control cho Chế độ hiển thị (Dark | Light) */
        section[data-testid="stSidebar"] div[role="radiogroup"]:has(label + label):not(:has(label + label + label)) {{
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            gap: 4px !important;
            background-color: var(--gen-surface-alt) !important;
            border: 1px solid var(--gen-control-border) !important;
            border-radius: 8px !important;
            padding: 3px !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]:has(label + label):not(:has(label + label + label)) > label {{
            border-left: 1px solid transparent !important;
            border-radius: 6px !important;
            padding: 6px 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]:has(label + label):not(:has(label + label + label)) > label:has(input:checked) {{
            background-color: var(--gen-surface) !important;
            border: 1px solid var(--gen-primary) !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"]:has(label + label):not(:has(label + label + label)) > label:has(input:checked) p {{
            color: var(--gen-text-primary) !important;
            font-weight: 600 !important;
        }}

        /* 4. KHẮC PHỤC TRIỆT ĐỂ HỘP CHỌN SELECTBOX / MULTISELECT BỊ ĐEN TRONG LIGHT MODE */
        .stSelectbox label, .stMultiSelect label, .stSlider label,
        .stDateInput label, .stNumberInput label, .stTextInput label,
        .stRadio label, .stCheckbox label, .stFileUploader label {{
            color: var(--gen-text-secondary) !important;
            font-size: 12.5px !important;
            font-weight: 500 !important;
        }}

        /* Ép màu sáng cho toàn bộ các lớp div lồng nhau bên trong Selectbox & Input */
        div[data-testid="stSelectbox"] div[data-baseweb="select"],
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"],
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"],
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        div[data-testid="stDateInput"] > div > div,
        div[data-testid="stNumberInput"] > div > div {{
            background-color: var(--gen-surface-alt) !important;
            color: var(--gen-text-primary) !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] {{
            border: 1px solid var(--gen-control-border) !important;
            border-radius: 6px !important;
            min-height: 38px !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] input,
        div[data-baseweb="input"] input,
        div[data-testid="stDateInput"] input,
        div[data-testid="stNumberInput"] input {{
            color: var(--gen-text-primary) !important;
            -webkit-text-fill-color: var(--gen-text-primary) !important;
        }}

        div[data-baseweb="select"] svg {{
            fill: var(--gen-text-secondary) !important;
            color: var(--gen-text-secondary) !important;
        }}

        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {{
            border-color: var(--gen-primary) !important;
            box-shadow: 0 0 0 2px var(--gen-primary-bg) !important;
        }}

        /* Menu xổ xuống khi bấm vào Selectbox */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="menu"],
        ul[role="listbox"],
        ul[data-testid="stSelectboxVirtualDropdown"] {{
            background-color: var(--gen-surface) !important;
            border: 1px solid var(--gen-control-border) !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12) !important;
        }}
        ul[role="listbox"] li,
        div[role="option"] {{
            color: var(--gen-text-primary) !important;
            background-color: var(--gen-surface) !important;
            font-size: 13px !important;
        }}
        ul[role="listbox"] li:hover,
        div[role="option"]:hover,
        ul[role="listbox"] li[aria-selected="true"],
        div[role="option"][aria-selected="true"] {{
            background-color: var(--gen-primary-bg) !important;
            color: var(--gen-primary-fg) !important;
        }}

        /* Thẻ Tag trong Multiselect (Xanh băng dịu mắt #E0F2FE / chữ #0369A1 ở Option 1) */
        span[data-baseweb="tag"] {{
            background-color: var(--gen-primary-bg) !important;
            color: var(--gen-primary-fg) !important;
            border: 1px solid var(--gen-primary-bd) !important;
            border-radius: 5px !important;
            padding: 2px 8px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
        }}
        span[data-baseweb="tag"] span,
        span[data-baseweb="tag"] svg {{
            color: var(--gen-primary-fg) !important;
            fill: var(--gen-primary-fg) !important;
        }}

        /* 5. KHẮC PHỤC THẺ <code> BỊ ĐEN & THANH CÔNG CỤ PLOTLY (.modebar) BỊ ĐEN */
        code, pre {{
            background-color: var(--gen-surface-alt) !important;
            color: var(--gen-primary-fg) !important;
            border: 1px solid var(--gen-control-border) !important;
            border-radius: 4px !important;
            padding: 1px 6px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 12px !important;
        }}

        .js-plotly-plot .plotly .modebar {{
            background-color: transparent !important;
        }}
        .js-plotly-plot .plotly .modebar-group {{
            background-color: var(--gen-surface-alt) !important;
            border: 1px solid var(--gen-border) !important;
            border-radius: 6px !important;
            padding: 2px 4px !important;
        }}
        .js-plotly-plot .plotly .modebar-btn svg path {{
            fill: var(--gen-text-secondary) !important;
        }}
        .js-plotly-plot .plotly .modebar-btn:hover svg path {{
            fill: var(--gen-primary) !important;
        }}

        /* 6. Tiêu đề màn hình & Khung thẻ */
        .section-header-wrap {{
            text-align: left !important;
            margin-bottom: 14px;
        }}
        .section-main-title {{
            font-family: 'General Sans', sans-serif !important;
            font-size: 23px !important;
            font-weight: 600 !important;
            color: var(--gen-text-primary) !important;
            margin: 0 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.25 !important;
        }}
        .section-desc {{
            font-size: 13.5px !important;
            color: var(--gen-text-secondary) !important;
            margin: 4px 0 0 0 !important;
            max-width: 920px;
            line-height: 1.5 !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: var(--gen-surface) !important;
            border: 1px solid var(--gen-border) !important;
            border-radius: 8px !important;
            padding: 16px 18px !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] h4 {{
            font-family: 'General Sans', sans-serif !important;
            font-size: 15.5px !important;
            font-weight: 600 !important;
            color: var(--gen-text-primary) !important;
            letter-spacing: -0.015em !important;
            margin: 0 0 4px 0 !important;
            padding: 0 !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] .stCaption,
        div[data-testid="stCaptionContainer"] p {{
            color: var(--gen-text-faint) !important;
            font-size: 12.5px !important;
        }}

        /* 7. Thẻ KPI 4 cột ngang (Hiển thị đầy đủ 100% tiêu đề không bị cắt '...') */
        .opt2-kpi-grid {{
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
            gap: 14px !important;
            width: 100% !important;
            margin-bottom: 16px !important;
        }}
        @media (max-width: 1100px) {{
            .opt2-kpi-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }}
        }}
        @media (max-width: 640px) {{
            .opt2-kpi-grid {{
                grid-template-columns: 1fr !important;
            }}
        }}
        .opt2-bento-card {{
            background: var(--gen-surface) !important;
            border: 1px solid var(--gen-border) !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            min-width: 0 !important;
        }}
        .opt2-card-label {{
            font-size: 12.5px !important;
            font-weight: 600 !important;
            color: var(--gen-text-secondary) !important;
            line-height: 1.35 !important;
            white-space: normal !important;
            overflow: visible !important;
            margin-bottom: 6px !important;
        }}
        .opt2-card-value {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 22px !important;
            font-weight: 600 !important;
            color: var(--gen-text-primary) !important;
            letter-spacing: -0.02em !important;
            font-variant-numeric: tabular-nums !important;
            line-height: 1.2 !important;
            margin: 8px 0 10px 0 !important;
        }}
        .opt2-card-footer {{
            border-top: 1px solid var(--gen-rule) !important;
            padding-top: 8px !important;
            font-size: 12px !important;
            color: var(--gen-text-faint) !important;
            line-height: 1.35 !important;
        }}

        /* 8. Bảng Tài chính Chuẩn Hóa */
        .gen-table-wrap {{
            width: 100%;
            overflow-x: auto;
            background: transparent;
            margin-top: 4px;
        }}
        .gen-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .gen-table thead tr {{
            background-color: var(--gen-surface-alt) !important;
            border-bottom: 1px solid var(--gen-control-border);
        }}
        .gen-table th {{
            padding: 9px 12px;
            font-family: 'DM Sans', sans-serif;
            font-size: 12px;
            font-weight: 600;
            color: var(--gen-text-secondary) !important;
            white-space: nowrap;
            text-align: left;
        }}
        .gen-table th.num-col {{
            text-align: right;
        }}
        .gen-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--gen-border);
            color: var(--gen-text-secondary) !important;
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            white-space: nowrap;
        }}
        .gen-table td:first-child {{
            color: var(--gen-text-primary) !important;
            font-weight: 600;
        }}
        .gen-table td.mono-col {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 12.5px;
            font-variant-numeric: tabular-nums;
            color: var(--gen-text-secondary) !important;
        }}
        .gen-table td.num-col {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 12.5px;
            font-variant-numeric: tabular-nums;
            text-align: right;
            color: var(--gen-text-primary) !important;
        }}
        .gen-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        .gen-table tbody tr:hover {{
            background-color: var(--gen-surface-alt);
        }}

        .delta-pos {{ color: var(--gen-success) !important; font-weight: 600; }}
        .delta-neg {{ color: var(--gen-error) !important; font-weight: 600; }}
        .delta-warn {{ color: var(--gen-warning) !important; font-weight: 600; }}

        /* 9. Huy hiệu trạng thái */
        .gen-chip {{
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px;
            font-weight: 600;
            line-height: 1.35;
            white-space: nowrap;
            width: fit-content;
        }}
        .chip-neutral {{
            background: var(--gen-surface-alt);
            color: var(--gen-text-secondary) !important;
            border: 1px solid var(--gen-control-border);
        }}
        .chip-teal, .chip-indigo {{
            background: var(--gen-primary-bg);
            color: var(--gen-primary-fg) !important;
            border: 1px solid var(--gen-primary-bd);
        }}
        .chip-success {{
            background: var(--gen-success-bg);
            color: var(--gen-success) !important;
            border: 1px solid var(--gen-success-bd);
        }}
        .chip-warning {{
            background: var(--gen-warning-bg);
            color: var(--gen-warning) !important;
            border: 1px solid var(--gen-warning-bd);
        }}
        .chip-error {{
            background: var(--gen-error-bg);
            color: var(--gen-error) !important;
            border: 1px solid var(--gen-error-bd);
        }}

        /* 10. Nút bấm */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 6px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            height: 36px !important;
            box-shadow: none !important;
            transition: background-color 140ms ease, border-color 140ms ease !important;
        }}
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
            background-color: {primary_btn_bg} !important;
            color: {primary_btn_fg} !important;
            border: 1px solid {primary_btn_bg} !important;
        }}
        .stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] * {{
            color: {primary_btn_fg} !important;
        }}
        .stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {{
            background-color: var(--gen-surface-alt) !important;
            color: var(--gen-text-primary) !important;
            border: 1px solid var(--gen-control-border) !important;
        }}
        .stButton > button[kind="secondary"] *, .stDownloadButton > button[kind="secondary"] * {{
            color: var(--gen-text-primary) !important;
        }}
        .stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {{
            background-color: var(--gen-surface-hover) !important;
            border-color: var(--gen-primary) !important;
        }}
    </style>
    """, unsafe_allow_html=True)