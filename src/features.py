"""TV3 Brent feature engineering for 1 and 5 trading-session horizons.

Each feature on row t uses observations available at t or earlier. Targets are
kept in the output for evaluation, while :func:`feature_columns` returns only
the 29 approved model inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR

PRICE = "DCOILBRENTEU"
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2024-12-31")
HORIZONS = (1, 5)
LAGS = (1, 2, 3, 5, 10, 20)
MA_WINDOWS = (5, 10, 20, 50)
VOL_WINDOWS = (5, 10, 20)
RET_WINDOWS = (5, 10, 20)

# Locked TV3 schema and order: 29 predictors, excluding date, targets, and split.
FEATURE_COLUMNS = (
    "price",
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10", "lag_20",
    "ret_1", "ret_5", "ret_10", "ret_20",
    "ret_1_lag1", "ret_1_lag2", "ret_1_lag3",
    "ma_5", "price_ma_5_ratio",
    "ma_10", "price_ma_10_ratio",
    "ma_20", "price_ma_20_ratio",
    "ma_50", "price_ma_50_ratio",
    "ma_5_20_ratio",
    "vol_5", "vol_10", "vol_20",
    "range_20", "dayofweek", "month",
)


def assign_split(target_date: pd.Series) -> pd.Series:
    """Assign rows using the date their label becomes observable."""
    out = pd.Series("", index=target_date.index, dtype=object)
    has_target = target_date.notna()
    out[has_target & (target_date <= TRAIN_END)] = "train"
    out[has_target & (target_date > TRAIN_END) & (target_date <= VAL_END)] = "val"
    out[has_target & (target_date > VAL_END)] = "test"
    return out


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build TV3's 29 predictors and 1/5-session labels from clean Brent data."""
    required = {"observation_date", PRICE}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing))}")

    df = raw[["observation_date", PRICE]].copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)
    p = df[PRICE]
    logp = np.log(p)

    f = pd.DataFrame({"observation_date": df["observation_date"], "price": p})

    # Price at t-k.
    for k in LAGS:
        f[f"lag_{k}"] = p.shift(k)

    # Log returns through t and lagged one-session returns.
    f["ret_1"] = logp.diff(1)
    for window in RET_WINDOWS:
        f[f"ret_{window}"] = logp.diff(window)
    for k in (1, 2, 3):
        f[f"ret_1_lag{k}"] = f["ret_1"].shift(k)

    # Trailing moving averages and price-to-average ratios.
    for window in MA_WINDOWS:
        moving_average = p.rolling(window).mean()
        f[f"ma_{window}"] = moving_average
        f[f"price_ma_{window}_ratio"] = p / moving_average - 1
    f["ma_5_20_ratio"] = f["ma_5"] / f["ma_20"] - 1

    # Trailing standard deviation of one-session log returns.
    for window in VOL_WINDOWS:
        f[f"vol_{window}"] = f["ret_1"].rolling(window).std()
    f["range_20"] = (p.rolling(20).max() - p.rolling(20).min()) / p.rolling(20).mean()

    f["dayofweek"] = f["observation_date"].dt.day_of_week
    f["month"] = f["observation_date"].dt.month

    # Future-looking values are labels only and are excluded from model inputs.
    for horizon in HORIZONS:
        f[f"target_date_{horizon}"] = f["observation_date"].shift(-horizon)
        f[f"target_{horizon}"] = p.shift(-horizon)
        f[f"target_ret_{horizon}"] = logp.shift(-horizon) - logp
        f[f"split_{horizon}"] = assign_split(f[f"target_date_{horizon}"])

    # Discard only the rolling warm-up rows; retain final rows with missing labels.
    f = f.dropna(subset=list(FEATURE_COLUMNS))
    return f.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the approved model inputs in their fixed TV3 order."""
    missing = set(FEATURE_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Thiếu đặc trưng TV3: {', '.join(sorted(missing))}")
    return list(FEATURE_COLUMNS)


def check_no_leakage(raw: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Confirm that truncating raw data at t leaves features through t unchanged."""
    rng = np.random.default_rng(0)
    cols = feature_columns(feats)
    sample_size = min(15, len(feats))
    for i in rng.choice(np.arange(len(feats)), size=sample_size, replace=False):
        t = feats.loc[i, "observation_date"]
        dates = pd.to_datetime(raw["observation_date"])
        truncated = build_features(raw.loc[dates <= t])
        row = truncated[truncated["observation_date"] == t]
        assert len(row) == 1, t
        actual = row[cols].iloc[0].to_numpy(float)
        expected = feats.loc[i, cols].to_numpy(float)
        assert np.allclose(actual, expected, equal_nan=True), f"Ro ri thong tin tai {t.date()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROCESSED_DATA_DIR / "brent_clean.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DATA_DIR / "brent_features.csv"
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    feats = build_features(raw)
    check_no_leakage(raw, feats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.output, index=False, date_format="%Y-%m-%d")

    print(f"Da ghi {args.output}: {len(feats)} dong, {len(feature_columns(feats))} dac trung")
    print(f"Quan sat: {feats.observation_date.min().date()} -> {feats.observation_date.max().date()}")
    for horizon in HORIZONS:
        counts = feats[f"split_{horizon}"].replace("", "(bo)").value_counts().to_dict()
        print(f"h={horizon}:", counts)
    print("Kiem tra ro ri thong tin: DAT")


if __name__ == "__main__":
    main()
