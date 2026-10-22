"""Load the handed-off Brent historical data."""

from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_PATH

DATE_COLUMN = "observation_date"
PRICE_COLUMN = "DCOILBRENTEU"


def load_brent_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Return valid observations ordered by date, with date and price columns."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu Brent: {source}")

    data = pd.read_csv(source)
    missing = {DATE_COLUMN, PRICE_COLUMN}.difference(data.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing))}")

    data = data.loc[:, [DATE_COLUMN, PRICE_COLUMN]].copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[PRICE_COLUMN] = pd.to_numeric(data[PRICE_COLUMN], errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN, PRICE_COLUMN])
    data = data.sort_values(DATE_COLUMN).drop_duplicates(DATE_COLUMN).reset_index(drop=True)
    return data
