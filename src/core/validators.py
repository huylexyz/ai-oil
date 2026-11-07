import pandas as pd
from src.core.config import OFFICIAL_HORIZONS, REQUIRED_PRED_COLS

def validate_prediction_df(df: pd.DataFrame, filename: str) -> tuple[bool, str]:
    """Kiểm tra tệp CSV dự báo có đầy đủ 6 cột bắt buộc theo quy định hay không."""
    missing = REQUIRED_PRED_COLS - set(df.columns)
    if missing:
        return False, f"Tệp {filename} thiếu cột bắt buộc: {sorted(missing)}"

    if df[["actual", "predicted"]].isna().any().any():
        return False, f"Tệp {filename} chứa giá trị rỗng (NaN) ở cột actual hoặc predicted."

    if not set(df["horizon"].unique()).issubset(OFFICIAL_HORIZONS):
        return False, f"Tệp {filename} có giá trị horizon nằm ngoài {OFFICIAL_HORIZONS}."

    return True, "Hợp lệ"


def validate_custom_inputs(
    price: float, lag_1: float, ma_5: float, ma_20: float, vol_5: float, ret_1_pct: float
) -> tuple[bool, list[str]]:
    """Kiểm tra lỗi nhập liệu khi người dùng giả lập thông số dự báo."""
    errors = []
    for label, val in [
        ("Giá hiện tại (price)", price),
        ("Giá trễ 1 phiên (lag_1)", lag_1),
        ("Trung bình trượt MA 5", ma_5),
        ("Trung bình trượt MA 20", ma_20),
    ]:
        if val <= 0:
            errors.append(f"{label} ({val}) phải lớn hơn 0 USD/thùng.")
        elif val > 300:
            errors.append(f"{label} ({val}) vượt ngưỡng giới hạn thực tế (300 USD/thùng).")

    if vol_5 < 0:
        errors.append("Độ biến động (vol_5) không được nhận giá trị âm.")
    elif vol_5 > 1.0:
        errors.append("Độ biến động (vol_5) vượt ngưỡng 1.0, vui lòng kiểm tra lại đơn vị.")

    if abs(ret_1_pct) > 50:
        errors.append(f"Mức thay đổi 1 phiên ({ret_1_pct:.2f}%) vượt quá biên độ ±50%/phiên.")

    return len(errors) == 0, errors
