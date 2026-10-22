"""Shared paths and interface conventions for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "DCOILBRENTEU_2015-2026.csv"
CURRENT_DATA_PATH = PROCESSED_DATA_DIR / "brent_observations.csv"
MODEL_ARTIFACT_PATH = MODEL_DIR / "forecast_models.joblib"

FORECAST_HORIZONS = (1, 5)
SPLIT_BOUNDARIES = {
    "train_end": "2022-12-31",
    "validation_start": "2023-01-01",
    "validation_end": "2024-12-31",
    "test_start": "2025-01-01",
    "test_end": "2026-09-30",
}
FORECASTS_PATH = RESULTS_DIR / "forecasts.csv"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
RESULT_COLUMNS = (
    "model",
    "horizon",
    "origin_date",
    "target_date",
    "actual",
    "predicted",
    "split",
)
SPLITS = ("train", "validation", "test")
