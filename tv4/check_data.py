
from pathlib import Path
import pandas as pd

# Duong dan den file du lieu cua TV3
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "processed" / "brent_features.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Khong tim thay file: {DATA_FILE}"
    )

# Doc du lieu
df = pd.read_csv(DATA_FILE)

print("=== 1. KICH THUOC DU LIEU ===")
print("So dong:", len(df))
print("So cot:", len(df.columns))

print("\n=== 2. DANH SACH COT ===")
for col in df.columns:
    print(col)

print("\n=== 3. NAM DONG DAU TIEN ===")
print(df.head().to_string())

print("\n=== 4. GIA TRI THIEU ===")
print(df.isna().sum().to_string())

print("\n=== 5. CAC COT MUC TIEU ===")
for col in [
    "target_1", "target_5",
    "target_date_1", "target_date_5",
    "split_1", "split_5"
]:
    if col in df.columns:
        print(f"{col}: CO")
        if col.startswith("split_"):
            print(df[col].value_counts(dropna=False))
    else:
        print(f"{col}: CHUA TIM THAY")

print("\nKIEM TRA HOAN TAT")
