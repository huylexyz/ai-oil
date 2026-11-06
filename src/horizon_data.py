"""Prepare TV3 training data for horizons 2, 3 and 4.

Run ``python -m src.horizon_data`` from the repository root. The official
1/5-horizon feature CSV and the default ``build_features()`` stay unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR
from .features import FEATURE_COLUMNS, PRICE, assign_split, build_features

NEW_HORIZONS = (2, 3, 4)
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "brent_clean.csv"


def build_horizon_frame(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return approved TV3 predictors and observed labels for one new horizon.

    ``split_<h>`` follows the target date, as in the official TV3 data. Model
    owners must also purge overlapping boundary labels during tuning/final fit.
    """
    if type(horizon) is not int or horizon not in NEW_HORIZONS:
        raise ValueError(f"Horizon mới phải thuộc {NEW_HORIZONS}: {horizon!r}")
    required = {"observation_date", PRICE}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dữ liệu Brent thiếu cột: {sorted(missing)}")

    clean = raw.loc[:, ["observation_date", PRICE]].copy()
    clean["observation_date"] = pd.to_datetime(clean["observation_date"], errors="raise")
    clean[PRICE] = pd.to_numeric(clean[PRICE], errors="raise")
    if clean.empty or clean.isna().any().any():
        raise ValueError("Dữ liệu Brent rỗng hoặc có giá trị thiếu")
    if clean["observation_date"].duplicated().any():
        raise ValueError("Ngày quan sát Brent bị trùng")
    if not np.isfinite(clean[PRICE].to_numpy(dtype=float)).all() or not clean[PRICE].gt(0).all():
        raise ValueError("Giá Brent phải hữu hạn và dương")
    clean = clean.sort_values("observation_date").reset_index(drop=True)

    # Reuse the approved 29 predictors; its existing 1/5 labels are not exported.
    features = build_features(clean).loc[:, ["observation_date", *FEATURE_COLUMNS]]
    prices = clean[PRICE]
    log_prices = np.log(prices)
    date_col = f"target_date_{horizon}"
    price_col = f"target_{horizon}"
    return_col = f"target_ret_{horizon}"
    split_col = f"split_{horizon}"
    labels = pd.DataFrame({
        "observation_date": clean["observation_date"],
        date_col: clean["observation_date"].shift(-horizon),
        price_col: prices.shift(-horizon),
        return_col: log_prices.shift(-horizon) - log_prices,
    })
    labels[split_col] = assign_split(labels[date_col])

    frame = features.merge(labels, on="observation_date", how="left", validate="one_to_one")
    # The last h observations have no observed future label and cannot be used
    # for training or Test evaluation; never synthesize their target dates.
    frame = frame.dropna(subset=[date_col, price_col, return_col]).reset_index(drop=True)
    if frame.empty or frame.isna().any().any() or not frame[date_col].gt(frame["observation_date"]).all():
        raise ValueError(f"Nhãn horizon {horizon} không hợp lệ")
    if not frame[split_col].isin(("train", "val", "test")).all():
        raise ValueError(f"Phân tập horizon {horizon} không hợp lệ")
    return frame.loc[:, ["observation_date", *FEATURE_COLUMNS, date_col, price_col, return_col, split_col]]


def main() -> None:
    if not CLEAN_DATA_FILE.is_file():
        raise FileNotFoundError(f"Thiếu dữ liệu sạch TV2: {CLEAN_DATA_FILE}")
    raw = pd.read_csv(CLEAN_DATA_FILE)
    frames = {h: build_horizon_frame(raw, h) for h in NEW_HORIZONS}
    destinations = {h: PROCESSED_DATA_DIR / f"brent_features_h{h}.csv" for h in NEW_HORIZONS}
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Không ghi đè dữ liệu đã có: {existing}")

    for horizon, path in destinations.items():
        frames[horizon].to_csv(path, index=False, date_format="%Y-%m-%d")
        print(f"h={horizon}: {path} ({len(frames[horizon])} dòng)")


if __name__ == "__main__":
    main()
