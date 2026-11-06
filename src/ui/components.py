import re
import pandas as pd
import streamlit as st


def _clean_sentence_case(text: str) -> str:
    """Chuyển chuỗi IN HOA toàn bộ về dạng Sentence case, giữ nguyên các từ viết tắt kỹ thuật."""
    s = str(text).strip()
    if not s:
        return s
    lowered = s.lower()
    lowered = lowered[0].upper() + lowered[1:]
    acronyms = ["rmse", "mae", "da", "usd", "eia", "fred", "ma", "ai", "mlops", "csv", "xgb", "xgboost", "min", "max", "tb"]
    for ac in acronyms:
        lowered = re.sub(rf"\b{ac}\b", ac.upper(), lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\brandom forest\b", "Random Forest", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\bridge\b", "Ridge", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\bnaive\b", "Naive", lowered, flags=re.IGNORECASE)
    return lowered


def render_top_brand_bar():
    """Giữ hàm tương thích ngược, không tạo khoảng trống thừa ở đầu vùng chính."""
    return


def render_section_header(overline: str, title: str, description: str = ""):
    """Tiêu đề màn hình chuẩn App Screen (23px, gọn gàng, không hiển thị dòng đánh số 01 / ...)."""
    desc_html = f'<p class="section-desc">{description}</p>' if description else ""
    st.markdown(f"""
    <div class="section-header-wrap">
        <h1 class="section-main-title">{title}</h1>
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


def render_bento_kpis(cards: list[dict]):
    """
    4-Column Accent Bento Cards (Hiển thị đầy đủ 100% tiêu đề không bị cắt '...')
    - 4 thẻ độc lập nằm ngang trên 1 hàng (khóa cứng CSS Grid + Inline Style).
    - Mỗi thẻ có vạch màu định danh 3px bên trái.
    - Hàng 1: Nhãn chỉ số hiển thị đầy đủ 100% chiều ngang.
    - Hàng 2: Huy hiệu trạng thái (Badge).
    - Hàng 3: Số liệu chính (JetBrains Mono tabular-nums).
    - Hàng 4: Đường kẻ mảnh 1px + Dòng mô tả phụ (Meta) hiển thị đầy đủ.
    """
    accent_vars = [
        "var(--gen-primary, #0284C7)",
        "var(--t5-accent, #0D9488)",
        "var(--gen-warning, #D97706)",
        "var(--gen-success, #15803D)",
    ]
    tone_chip_map = {
        "success": "chip-success",
        "warning": "chip-warning",
        "error": "chip-error",
        "indigo": "chip-teal",
        "neutral": "chip-neutral",
    }

    cards_html = []
    for idx, c in enumerate(cards):
        clean_label = _clean_sentence_case(c.get("overline", ""))
        badge_text = str(c.get("badge", "")).strip()
        meta_text = str(c.get("meta", "")).strip()
        val_text = str(c.get("value", "")).strip()
        tone = c.get("tone", "neutral")
        chip_cls = tone_chip_map.get(tone, "chip-neutral")
        edge_color = accent_vars[idx % len(accent_vars)]

        badge_html = (
            f'<div style="margin-bottom: 6px;"><span class="gen-chip {chip_cls}">{badge_text}</span></div>'
            if badge_text else ""
        )

        card_item = (
            f'<div class="opt2-bento-card" style="'
            f'background: var(--gen-surface, #FFFFFF); '
            f'border: 1px solid var(--gen-border, #E2E8F0); '
            f'border-left: 3px solid {edge_color}; '
            f'border-radius: 8px; '
            f'padding: 14px 16px; '
            f'display: flex; '
            f'flex-direction: column; '
            f'justify-content: space-between; '
            f'min-width: 0;">'
            f'  <div>'
            f'    <div class="opt2-card-label" style="'
            f'font-size: 12.5px; font-weight: 600; color: var(--gen-text-secondary, #475569); '
            f'line-height: 1.35; margin-bottom: 6px; white-space: normal; overflow: visible;">{clean_label}</div>'
            f'    {badge_html}'
            f'  </div>'
            f'  <div class="opt2-card-value" style="'
            f'font-family: JetBrains Mono, monospace; font-size: 21px; font-weight: 600; '
            f'color: var(--gen-text-primary, #0F172A); letter-spacing: -0.02em; '
            f'font-variant-numeric: tabular-nums; line-height: 1.2; margin: 4px 0 10px 0;">{val_text}</div>'
            f'  <div class="opt2-card-footer" style="'
            f'border-top: 1px solid var(--gen-rule, #F1F5F9); padding-top: 8px; '
            f'font-size: 12px; color: var(--gen-text-faint, #64748B); '
            f'line-height: 1.35; white-space: normal;">{meta_text}</div>'
            f'</div>'
        )
        cards_html.append(card_item)

    grid_html = (
        f'<div class="opt2-kpi-grid" style="'
        f'display: grid; '
        f'grid-template-columns: repeat(4, minmax(0, 1fr)); '
        f'gap: 14px; '
        f'width: 100%; '
        f'margin-bottom: 16px;">'
        f'{"".join(cards_html)}'
        f'</div>'
    )
    st.markdown(grid_html, unsafe_allow_html=True)


def _is_numeric_column(series: pd.Series, col_name: str) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return col_name.lower() not in ["hạng", "rank"]
    sample = series.dropna().astype(str).head(5).tolist()
    if not sample:
        return False
    num_pattern = re.compile(r"^[\$\+\-]?\d+([.,]\d+)?%?(\s*\([\+\-].*\))?$")
    return all(bool(num_pattern.match(s.strip())) for s in sample)


def _format_cell_value(val, is_num: bool) -> str:
    text = str(val)
    if text in ["ĐẠT", "Sẵn sàng"]:
        return f'<span class="gen-chip chip-success">{text}</span>'
    if text in ["Đang chờ tệp", "LỖI RANH GIỚI"]:
        return f'<span class="gen-chip chip-warning">{text}</span>'

    if "(" in text and ")" in text and ("+" in text or "-" in text):
        prefix, rest = text.split("(", 1)
        delta_part = "(" + rest
        cls = "delta-pos" if "+" in delta_part else "delta-neg"
        return f'{prefix}<span class="{cls}">{delta_part}</span>'

    return text


def render_genesis_table(df: pd.DataFrame, max_height: int = None):
    """Bảng dữ liệu chuẩn Rules & Alignment: Tiêu đề rõ nét, số căn phải."""
    if df is None or df.empty:
        st.info("Không có dữ liệu hiển thị.")
        return

    num_flags = {col: _is_numeric_column(df[col], str(col)) for col in df.columns}

    headers_html = "".join([
        f'<th class="{"num-col" if num_flags[col] else ""}">{col}</th>'
        for col in df.columns
    ])

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            is_num = num_flags[col]
            raw_val = row[col]
            if isinstance(raw_val, float):
                formatted_str = f"{raw_val:,.4f}".rstrip("0").rstrip(".") if abs(raw_val) < 10 else f"{raw_val:,.2f}"
            else:
                formatted_str = str(raw_val)

            cell_content = _format_cell_value(formatted_str, is_num)
            is_date_or_code = any(k in str(col).lower() for k in ["ngày", "date", "mốc", "kỳ hạn", "trường", "công thức", "tệp"])
            td_class = "num-col" if is_num else ("mono-col" if is_date_or_code else "")
            cells.append(f'<td class="{td_class}">{cell_content}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    scroll_style = f'style="max-height: {max_height}px; overflow-y: auto;"' if max_height else ""
    table_html = (
        f'<div class="gen-table-wrap" {scroll_style}>'
        f'  <table class="gen-table">'
        f'    <thead><tr>{headers_html}</tr></thead>'
        f'    <tbody>{"".join(rows_html)}</tbody>'
        f'  </table>'
        f'</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_editorial_note(title: str, content: str):
    """Dòng chú thích gọn nhẹ dưới biểu đồ."""
    st.markdown(f"""
    <div style="padding: 6px 2px 18px 2px; font-size: 13px; color: var(--gen-text-faint); line-height: 1.5;">
        <strong style="color: var(--gen-text-secondary); font-weight: 600;">{title}:</strong> {content}
    </div>
    """, unsafe_allow_html=True)