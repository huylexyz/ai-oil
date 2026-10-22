"""Shared paths and interface conventions for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "DCOILBRENTEU_2015-2026.csv"

FORECAST_HORIZONS = (1, 5)
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
