from pathlib import Path
from src.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_DATA_PATH,
    FORECAST_HORIZONS,
)
from src.features import FEATURE_COLUMNS

BASE_DIR = PROJECT_ROOT
DATA_DIR = PROCESSED_DATA_DIR
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = PROJECT_ROOT / "tv4" / "predictions"

RAW_DATA_PATH = DEFAULT_DATA_PATH
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "brent_clean.csv"
REMOVED_ROWS_PATH = PROCESSED_DATA_DIR / "removed_rows.csv"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "brent_features.csv"

TRAIN_END_DATE = "2022-12-31"
VAL_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"

LAGS = [1, 2, 3, 5, 10, 20]
MAS = [5, 10, 20, 50]
VOLS = [5, 10, 20]
RETS = [1, 5, 10, 20]

MODELS_LIST = ["Naive", "Ridge", "Random Forest", "XGBoost"]
HORIZONS = list(FORECAST_HORIZONS)
FEATURE_COLS = list(FEATURE_COLUMNS)

MODEL_FILE_CANDIDATES = {
    ("Ridge", 1): [PROJECT_ROOT / "tv4" / "models" / "ridge_horizon_1.joblib"],
    ("Ridge", 5): [PROJECT_ROOT / "tv4" / "models" / "ridge_horizon_5.joblib"],
    ("Random Forest", 1): [PROJECT_ROOT / "tv5" / "models" / "rf_model_t1.joblib"],
    ("Random Forest", 5): [PROJECT_ROOT / "tv5" / "models" / "rf_model_t5.joblib"],
    ("XGBoost", 1): [PROJECT_ROOT / "tv6" / "models" / "xgb_horizon_1.joblib"],
    ("XGBoost", 5): [PROJECT_ROOT / "tv6" / "models" / "xgb_horizon_5.joblib"],
}

REQUIRED_PRED_COLS = {
    "observation_date",
    "target_date",
    "horizon",
    "actual",
    "predicted",
    "model",
}

FEATURE_GLOSSARY = [
    {"Nhóm đặc trưng": "Giá cơ sở & Độ trễ (Lags)", "Trường dữ liệu": "price, lag_1, lag_2, lag_3, lag_5, lag_10, lag_20", "Mô tả kỹ thuật": "Giá đóng cửa tại phiên quan sát t và các phiên liền trước (t-k)."},
    {"Nhóm đặc trưng": "Trung bình trượt (Moving Averages)", "Trường dữ liệu": "ma_5, ma_10, ma_20, ma_50", "Mô tả kỹ thuật": "Trung bình cộng giá của 5, 10, 20, 50 phiên gần nhất."},
    {"Nhóm đặc trưng": "Tỷ lệ hội tụ / phân kỳ MA", "Trường dữ liệu": "price_ma_5_ratio .. ma_5_20_ratio", "Mô tả kỹ thuật": "Khoảng cách tương đối giữa giá hiện tại và các đường trung bình động (P_t / MA_w - 1)."},
    {"Nhóm đặc trưng": "Tỷ suất sinh lời (Log-Returns)", "Trường dữ liệu": "ret_1, ret_5, ret_10, ret_20, ret_1_lag1..3", "Mô tả kỹ thuật": "Tốc độ biến thiên logarit ln(P_t / P_{t-w}) qua 1, 5, 10, 20 phiên và độ trễ."},
    {"Nhóm đặc trưng": "Độ biến động (Volatility & Range)", "Trường dữ liệu": "vol_5, vol_10, vol_20, range_20", "Mô tả kỹ thuật": "Độ lệch chuẩn của ret_1 trong 5, 10, 20 phiên và biên độ (Max20 - Min20)/MA20."},
    {"Nhóm đặc trưng": "Chu kỳ thời gian (Calendar)", "Trường dữ liệu": "dayofweek, month", "Mô tả kỹ thuật": "Thứ trong tuần (0–4) và tháng trong năm (1–12) của ngày quan sát."},
]

METRICS_GLOSSARY = [
    {"Chỉ số": "MAE (Mean Absolute Error)", "Công thức": "mean(|y - ŷ|)", "Diễn giải": "Sai số tuyệt đối trung bình (USD/thùng). Càng thấp càng tốt."},
    {"Chỉ số": "RMSE (Root Mean Squared Error)", "Công thức": "sqrt(mean((y - ŷ)²))", "Diễn giải": "Sai số bình phương trung bình căn bậc hai (USD/thùng). Càng thấp càng tốt."},
    {"Chỉ số": "Directional Accuracy (DA)", "Công thức": "mean(sign(y - P_obs) == sign(ŷ - P_obs)) × 100%", "Diễn giải": "Tỷ lệ phần trăm dự báo đúng chiều tăng/giảm so với giá tại observation_date (Naive không xác định chiều nên hiển thị N/A)."},
]