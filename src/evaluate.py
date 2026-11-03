"""Evaluate handed-off Brent forecasts by model, horizon, and split."""

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR

INPUT_COLUMNS = (
    "observation_date", "target_date", "horizon", "actual",
    "predicted", "model", "split",
)
METRIC_COLUMNS = (
    "model", "horizon", "split", "n_samples", "mae", "rmse",
    "directional_accuracy",
)
BASE_PRICE_FILE = PROCESSED_DATA_DIR / "brent_clean.csv"
BASE_DATE = "observation_date"
BASE_PRICE = "DCOILBRENTEU"


def _base_prices() -> pd.DataFrame:
    if not BASE_PRICE_FILE.is_file():
        raise FileNotFoundError(f"Thiếu dữ liệu sạch TV2: {BASE_PRICE_FILE}")
    prices = pd.read_csv(BASE_PRICE_FILE)
    missing = {BASE_DATE, BASE_PRICE}.difference(prices.columns)
    if missing:
        raise ValueError(f"Dữ liệu TV2 thiếu cột {sorted(missing)}: {BASE_PRICE_FILE}")
    prices = prices.loc[:, [BASE_DATE, BASE_PRICE]].copy()
    prices[BASE_DATE] = pd.to_datetime(prices[BASE_DATE], errors="raise")
    prices[BASE_PRICE] = pd.to_numeric(prices[BASE_PRICE], errors="raise")
    if prices.isna().any().any() or not np.isfinite(prices[BASE_PRICE].to_numpy()).all():
        raise ValueError("Dữ liệu TV2 có ngày hoặc giá gốc không hợp lệ")
    if prices[BASE_DATE].duplicated().any():
        raise ValueError("Dữ liệu TV2 có ngày quan sát trùng")
    return prices.rename(columns={BASE_PRICE: "origin_price"})


def calculate_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate unrounded USD/barrel errors and directional accuracy (0–100%).

    A zero change is a third direction: it is correct only when both actual and
    predicted changes are zero. Naive persistence has undefined directional
    accuracy and is returned as NaN. An empty input returns an empty table.
    """
    if not isinstance(backtest_df, pd.DataFrame):
        raise TypeError("backtest_df phải là pandas DataFrame")
    missing = set(INPUT_COLUMNS).difference(backtest_df.columns)
    if missing:
        raise ValueError(f"Backtesting thiếu cột {sorted(missing)}")
    if backtest_df.empty:
        return pd.DataFrame({
            "model": pd.Series(dtype="str"),
            "horizon": pd.Series(dtype="int64"),
            "split": pd.Series(dtype="str"),
            "n_samples": pd.Series(dtype="int64"),
            "mae": pd.Series(dtype="float64"),
            "rmse": pd.Series(dtype="float64"),
            "directional_accuracy": pd.Series(dtype="float64"),
        }).loc[:, list(METRIC_COLUMNS)]

    frame = backtest_df.loc[:, list(INPUT_COLUMNS)].copy()
    for column in ("observation_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in ("actual", "predicted", "horizon"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any():
        raise ValueError("Backtesting có giá trị thiếu")
    if not np.isfinite(frame[["actual", "predicted", "horizon"]].to_numpy(dtype=float)).all():
        raise ValueError("Backtesting có giá trị số không hữu hạn")
    if not frame["horizon"].gt(0).all() or not frame["horizon"].eq(frame["horizon"].astype("int64")).all():
        raise ValueError("horizon phải là số nguyên dương")
    frame["horizon"] = frame["horizon"].astype("int64")
    if not frame["target_date"].gt(frame["observation_date"]).all():
        raise ValueError("target_date phải sau observation_date")
    if not frame["model"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("model phải là chuỗi không rỗng")
    if not frame["split"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("split phải là chuỗi không rỗng")

    frame = frame.merge(_base_prices(), on="observation_date", how="left", validate="many_to_one")
    if frame["origin_price"].isna().any():
        dates = frame.loc[frame["origin_price"].isna(), "observation_date"]
        examples = ", ".join(dates.dt.strftime("%Y-%m-%d").drop_duplicates().head(5))
        raise ValueError(f"Không tìm thấy giá TV2 tại observation_date: {examples}")

    rows = []
    for (model, horizon, split), group in frame.groupby(
        ["model", "horizon", "split"], sort=False
    ):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        origin = group["origin_price"].to_numpy(dtype=float)
        error = actual - predicted
        accuracy = (
            np.nan if model == "Naive" else
            100.0 * np.mean(np.sign(actual - origin) == np.sign(predicted - origin))
        )
        rows.append({
            "model": model,
            "horizon": horizon,
            "split": split,
            "n_samples": len(group),
            "mae": np.mean(np.abs(error)),
            "rmse": np.sqrt(np.mean(np.square(error))),
            "directional_accuracy": accuracy,
        })
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)
