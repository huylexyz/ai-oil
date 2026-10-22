"""Run the reproducible Brent forecasting experiment and save its outputs."""

import json
import pandas as pd
import sys

from .config import FORECASTS_PATH, METRICS_PATH, SPLIT_BOUNDARIES
from .data_loader import DATE_COLUMN, PRICE_COLUMN, DEFAULT_DATA_PATH, load_brent_data
from .evaluate import evaluate
from .features import build_features, feature_columns
from .predict import build_forecasts


def run_experiment(data_path=DEFAULT_DATA_PATH):
    raw = pd.read_csv(data_path)
    data = load_brent_data(data_path)
    if len(data) != 2982:
        raise ValueError(f"Kỳ vọng 2.982 quan sát hợp lệ, nhận được {len(data):,}.")

    features = build_features(data)
    split_counts = {}
    for horizon in (1, 5):
        dates = pd.to_datetime(features[f"target_date_{horizon}"])
        split_counts[str(horizon)] = {
            "train": int((dates <= pd.Timestamp(SPLIT_BOUNDARIES["train_end"])).sum()),
            "validation": int(
                dates.between(
                    SPLIT_BOUNDARIES["validation_start"],
                    SPLIT_BOUNDARIES["validation_end"],
                ).sum()
            ),
            "test": int(
                dates.between(
                    SPLIT_BOUNDARIES["test_start"], SPLIT_BOUNDARIES["test_end"]
                ).sum()
            ),
        }
    forecasts = build_forecasts(features, feature_columns())
    metrics = evaluate(forecasts, features)
    FORECASTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(FORECASTS_PATH, index=False, date_format="%Y-%m-%d")
    metrics.to_csv(METRICS_PATH, index=False)

    # Share observation counts and split boundaries with the dashboard.
    raw_dates = pd.to_datetime(raw[DATE_COLUMN], errors="coerce")
    numeric_prices = pd.to_numeric(raw[PRICE_COLUMN], errors="coerce")
    report = {
        "source": str(data_path),
        "raw_rows": int(len(raw)),
        "valid_observations": int(len(data)),
        "missing_or_non_numeric_prices": int(numeric_prices.isna().sum()),
        "invalid_dates": int(raw_dates.isna().sum()),
        "duplicate_dates": int(raw_dates.duplicated().sum()),
        "first_valid_date": data[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "last_valid_date": data[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "split_boundaries": SPLIT_BOUNDARIES,
        "target_observations_by_horizon": split_counts,
        "features": feature_columns(),
        "missing_value_policy": "Drop rows without a valid date or numeric price; do not interpolate.",
        "scaling": "StandardScaler fitted inside Ridge pipeline on each fit split only.",
        "test_policy": "Fixed model settings; fit on train+validation; test is evaluation-only.",
    }
    (METRICS_PATH.parent / "data_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return forecasts, metrics, report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    forecasts, metrics, report = run_experiment()
    print(
        f"{report['valid_observations']:,} quan sát hợp lệ; "
        f"{len(forecasts):,} dự báo; kết quả: {METRICS_PATH}"
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
