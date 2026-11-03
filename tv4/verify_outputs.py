
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

required_columns = [
    "observation_date",
    "target_date",
    "horizon",
    "actual",
    "predicted",
    "model",
]

files = [
    ("predictions/naive_horizon_1.csv", 1, "Naive"),
    ("predictions/naive_horizon_5.csv", 5, "Naive"),
    ("predictions/ridge_horizon_1.csv", 1, "Ridge"),
    ("predictions/ridge_horizon_5.csv", 5, "Ridge"),
]

print("=== KIEM TRA CSV ===")

for relative_path, horizon, model_name in files:
    path = BASE_DIR / relative_path

    if not path.exists():
        print(f"THIEU FILE: {relative_path}")
        continue

    df = pd.read_csv(path)
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    print(f"\n{relative_path}")
    print("So dong:", len(df))
    print("Thieu cot:", missing if missing else "Khong")
    print("Horizon dung:", df["horizon"].eq(horizon).all()
          if "horizon" in df.columns else False)
    print("Model dung:", df["model"].eq(model_name).all()
          if "model" in df.columns else False)

    if "split" in df.columns:
        print("Phan chia:")
        print(df["split"].value_counts().to_dict())

    if {"actual", "predicted"}.issubset(df.columns):
        print("Gia tri du bao bi thieu:",
              df["predicted"].isna().sum())

print("\n=== KIEM TRA NAP MO HINH ===")

for horizon in [1, 5]:
    path = BASE_DIR / "models" / f"ridge_horizon_{horizon}.joblib"

    try:
        saved = joblib.load(path)
        print(
            f"Horizon {horizon}: OK | "
            f"alpha={saved['alpha']} | "
            f"features={len(saved['features'])}"
        )
    except Exception as exc:
        print(f"Horizon {horizon}: LOI - {exc}")

print("\nKIEM TRA HOAN TAT")
