
from pathlib import Path
import pandas as pd

# Thu muc du an
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "processed" / "brent_features.csv"
OUTPUT_DIR = BASE_DIR / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Doc du lieu TV3
df = pd.read_csv(DATA_FILE)

def create_naive_predictions(df, horizon):
    """
    horizon = 1: du bao 1 phien
    horizon = 5: du bao 5 phien
    """
    split_col = f"split_{horizon}"
    target_col = f"target_{horizon}"
    target_date_col = f"target_date_{horizon}"

    # Chi lay cac dong co nhan va split hop le
    valid = df[
        df[split_col].isin(["train", "val", "test"])
        & df[target_col].notna()
        & df[target_date_col].notna()
    ].copy()

    # Naive: gia du bao bang gia hien tai
    result = pd.DataFrame({
        "observation_date": valid["observation_date"],
        "target_date": valid[target_date_col],
        "horizon": horizon,
        "actual": valid[target_col],
        "predicted": valid["price"],
        "model": "Naive",
        "split": valid[split_col],
    })

    return result


# Tao du bao cho ca hai horizon
pred_1 = create_naive_predictions(df, 1)
pred_5 = create_naive_predictions(df, 5)

# Luu ket qua
pred_1.to_csv(
    OUTPUT_DIR / "naive_horizon_1.csv",
    index=False
)

pred_5.to_csv(
    OUTPUT_DIR / "naive_horizon_5.csv",
    index=False
)

print("Naive Forecast hoan tat!")
print("Horizon 1:", len(pred_1), "dong")
print("Horizon 5:", len(pred_5), "dong")

print("\nSo dong theo split - Horizon 1:")
print(pred_1["split"].value_counts())

print("\nSo dong theo split - Horizon 5:")
print(pred_5["split"].value_counts())

print("\nVi du ket qua Horizon 1:")
print(pred_1.head().to_string(index=False))

print("\nDa luu file tai:", OUTPUT_DIR)
