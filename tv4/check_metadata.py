
import joblib
import sklearn
from pathlib import Path

print("Scikit-learn version:", sklearn.__version__)

model_data = joblib.load(Path(__file__).resolve().parent / "models" / "ridge_horizon_5.joblib")

print("\n=== RIDGE HORIZON 5 METADATA ===")

for key, value in model_data.items():
    if key != "model":
        print(f"{key}: {value}")
