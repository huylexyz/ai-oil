"""Create chronologically separated forecasts for the supported horizons."""

import pandas as pd

from .config import FORECAST_HORIZONS, RESULT_COLUMNS, SPLIT_BOUNDARIES


def split_for_date(value) -> str | None:
    """Assign a target date to the requested train/validation/test window."""
    date = pd.Timestamp(value)
    if date <= pd.Timestamp(SPLIT_BOUNDARIES["train_end"]):
        return "train"
    if pd.Timestamp(SPLIT_BOUNDARIES["validation_start"]) <= date <= pd.Timestamp(
        SPLIT_BOUNDARIES["validation_end"]
    ):
        return "validation"
    if pd.Timestamp(SPLIT_BOUNDARIES["test_start"]) <= date <= pd.Timestamp(
        SPLIT_BOUNDARIES["test_end"]
    ):
        return "test"
    return None


def build_forecasts(features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Evaluate validation using train fit, then test using train+validation fit.

    Model settings are fixed in ``train.py``. Validation is reported as a
    chronological holdout; the test window is never used to fit or tune.
    """
    from .train import fit_models

    records = []
    for horizon in FORECAST_HORIZONS:
        target_date_col = f"target_date_{horizon}"
        target_col = f"target_{horizon}"
        frame = features.copy()
        frame = frame.dropna(subset=[target_col, target_date_col]).copy()
        frame["split"] = frame[target_date_col].map(split_for_date)

        for evaluated_split in ("validation", "test"):
            fit_splits = ["train"] if evaluated_split == "validation" else ["train", "validation"]
            fit_rows = frame[frame["split"].isin(fit_splits)]
            eval_rows = frame[frame["split"] == evaluated_split]
            if fit_rows.empty or eval_rows.empty:
                continue

            fitted = fit_models(fit_rows[feature_names], fit_rows[target_col])
            predictions = {
                "Naive Persistence": eval_rows["actual"].to_numpy(),
                **{
                    name: model.predict(eval_rows[feature_names])
                    for name, model in fitted.items()
                },
            }
            for model_name, values in predictions.items():
                for (_, row), predicted in zip(eval_rows.iterrows(), values):
                    records.append(
                        {
                            "model": model_name,
                            "horizon": horizon,
                            "origin_date": row["origin_date"],
                            "target_date": row[target_date_col],
                            "actual": float(row[target_col]),
                            # Tree ensembles can vary by sub-picounit roundoff
                            # across parallel reductions; cents-level inputs
                            # do not justify preserving that numeric noise.
                            "predicted": round(float(predicted), 10),
                            "split": evaluated_split,
                        }
                    )

    return pd.DataFrame(records, columns=RESULT_COLUMNS)
