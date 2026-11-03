
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
features = pd.read_csv(base.parent / "data" / "processed" / "brent_features.csv")

checks = [
    ("naive_horizon_1.csv", 1),
    ("naive_horizon_5.csv", 5),
    ("ridge_horizon_1.csv", 1),
    ("ridge_horizon_5.csv", 5),
]

for filename, horizon in checks:
    pred = pd.read_csv(base / "predictions" / filename)

    date_col = f"target_date_{horizon}"
    split_col = f"split_{horizon}"

    expected = features[
        features[split_col].isin(["train", "val", "test"])
        & features[f"target_{horizon}"].notna()
    ]

    merged = pred.merge(
        expected[[
            "observation_date",
            date_col,
            f"target_{horizon}",
        ]],
        on="observation_date",
        how="left",
        validate="one_to_one",
    )

    date_ok = (
        pd.to_datetime(merged["target_date"])
        == pd.to_datetime(merged[date_col])
    ).all()

    actual_ok = (
        (merged["actual"] - merged[f"target_{horizon}"])
        .abs()
        .lt(1e-8)
    ).all()

    print(f"\n{filename}")
    print("Ngay muc tieu khop:", date_ok)
    print("Gia thuc te khop:", actual_ok)
