from pathlib import Path

import numpy as np
import pandas as pd


# Đường dẫn tính từ vị trí file, không phụ thuộc thư mục chạy lệnh.
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "brent_features.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Khong tim thay: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required = [
    "observation_date", "price",
    "target_1", "target_date_1", "split_1",
    "target_5", "target_date_5", "split_5",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Thieu cot: {missing}")

for col in ["observation_date", "target_date_1", "target_date_5"]:
    df[col] = pd.to_datetime(df[col], errors="raise")

assert df["observation_date"].is_unique, "Co ngay quan sat trung!"
assert df["observation_date"].is_monotonic_increasing, "Ngay chua tang dan!"

features = [
    c for c in df.columns
    if c != "observation_date"
    and not c.startswith(("target_", "split_"))
]

assert len(features) == 29, f"Can 29 dac trung, hien co {len(features)}"
assert all(
    pd.api.types.is_numeric_dtype(df[c]) for c in features
), "Co dac trung khong phai so!"
assert np.isfinite(df[features].to_numpy(dtype=float)).all(), (
    "Dac trung co gia tri thieu hoac vo cuc!"
)

print("FILE:", DATA_FILE)
print("So dong:", len(df))
print("So cot:", len(df.columns))
print("So dac trung:", len(features))
print("Tu ngay:", df["observation_date"].min().date())
print("Den ngay:", df["observation_date"].max().date())
print("\nDAC TRUNG:")
print(", ".join(features))

for h in (1, 5):
    target = f"target_{h}"
    target_date = f"target_date_{h}"
    split = f"split_{h}"

    valid = df[df[split].isin(["train", "val", "test"])].copy()
    assert valid[[target, target_date]].notna().all().all(), (
        f"Horizon {h}: dong co split nhung thieu nhan!"
    )
    assert (valid[target_date] > valid["observation_date"]).all(), (
        f"Horizon {h}: ngay muc tieu khong sau ngay quan sat!"
    )

    train = valid[valid[split] == "train"].copy()
    val = valid[valid[split] == "val"].copy()
    test = valid[valid[split] == "test"].copy()

    assert min(len(train), len(val), len(test)) > 0, "Co tap rong!"
    assert (train[target_date] <= pd.Timestamp("2022-12-31")).all()
    assert val[target_date].between("2023-01-01", "2024-12-31").all()
    assert test[target_date].between("2025-01-01", "2026-09-30").all()

    # Giả định dự báo sau khi đã biết giá của phiên quan sát.
    # Nhãn Train phải có sẵn ở lần dự báo Validation đầu tiên.
    cutoff = val["observation_date"].min()
    safe_train = train[train[target_date] <= cutoff]

    print(f"\nHORIZON {h}")
    print("Train ban dau:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))
    print("Dong cuoi chua co nhan:", int(df[target].isna().sum()))
    print("Ngay du bao Validation dau tien:", cutoff.date())
    print("Mau Train can loai tai ranh gioi:", len(train) - len(safe_train))
    print("Train an toan:", len(safe_train))

print("\nKIEM TRA DU LIEU: DAT")