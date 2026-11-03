from pathlib import Path

import joblib
import matplotlib

# Xuất ảnh trực tiếp, không cần mở cửa sổ đồ họa.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"


def analyze_horizon(horizon):
    model_path = MODEL_DIR / f"xgb_horizon_{horizon}.joblib"
    artifact = joblib.load(model_path)

    model = artifact["model"]
    features = artifact["features"]

    assert artifact["horizon"] == horizon
    assert model.importance_type == "gain"
    assert len(features) == model.n_features_in_

    values = model.feature_importances_

    assert len(values) == len(features)
    assert np.isfinite(values).all()

    importance = pd.DataFrame({
        "feature": features,
        "importance_gain": values,
    }).sort_values(
        ["importance_gain", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    importance.insert(0, "rank", range(1, len(importance) + 1))

    csv_path = RESULT_DIR / f"xgb_importance_h{horizon}.csv"
    importance.to_csv(csv_path, index=False)

    # Đảo thứ tự để đặc trưng quan trọng nhất nằm trên cùng.
    top10 = importance.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top10["feature"],
        top10["importance_gain"],
        color="#2878B5",
    )
    ax.set_title(
        f"XGBoost: Top 10 feature importances — horizon {horizon}"
    )
    ax.set_xlabel("Normalized gain importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()

    image_path = RESULT_DIR / f"xgb_importance_h{horizon}.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nHORIZON {horizon}")
    print(
        importance.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(
        "So dac trung co importance > 0:",
        int((importance["importance_gain"] > 0).sum()),
    )
    print("CSV:", csv_path)
    print("Bieu do:", image_path)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in (1, 5):
        analyze_horizon(horizon)

    print("\nHOAN TAT PHAN TICH DAC TRUNG")


if __name__ == "__main__":
    main()