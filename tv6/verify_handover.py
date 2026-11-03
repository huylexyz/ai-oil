from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT.parent / "data/processed/brent_features.csv"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    required = [
        "README.md",
        "requirements.txt",
        "check_data.py",
        "train_xgb.py",
        "tune_xgb.py",
        "finalize_xgb.py",
        "analyze_importance.py",
        "results/xgb_summary.json",
        "results/xgb_predictions_test.csv",
    ]

    for h in (1, 5):
        required.extend([
            f"models/xgb_horizon_{h}.joblib",
            f"results/xgb_predictions_test_h{h}.csv",
            f"results/xgb_importance_h{h}.csv",
            f"results/xgb_importance_h{h}.png",
            f"results/tuning/xgb_tuning_h{h}.csv",
            f"results/tuning/xgb_tuning_summary_h{h}.json",
        ])

    for name in required:
        path = ROOT / name
        check(path.is_file(), f"Thieu file: {name}")
        check(path.stat().st_size > 0, f"File rong: {name}")

    print("Danh sach file: DAT")

    df = pd.read_csv(DATA_FILE)
    summary = json.loads(
        (ROOT / "results/xgb_summary.json").read_text(encoding="utf-8")
    )
    data_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    check(
        data_hash == summary["data_sha256"],
        "Du lieu TV3 da thay doi sau khi huan luyen!",
    )
    print("Hash du lieu TV3: DAT")

    expected_columns = [
        "observation_date", "target_date", "horizon",
        "actual", "predicted", "model", "split",
    ]
    parts = []

    for h in (1, 5):
        pred = pd.read_csv(
            ROOT / f"results/xgb_predictions_test_h{h}.csv"
        )
        source = df[df[f"split_{h}"] == "test"].copy()

        check(
            list(pred.columns) == expected_columns,
            f"Horizon {h}: sai dinh dang cot",
        )
        check(
            len(pred) == len(source) == 442,
            f"Horizon {h}: sai so dong Test",
        )
        check(
            not pred.isna().any().any(),
            f"Horizon {h}: du bao co gia tri thieu",
        )
        check(
            not pred.duplicated(
                ["observation_date", "target_date", "horizon"]
            ).any(),
            f"Horizon {h}: trung mau du bao",
        )
        check(
            pred["horizon"].eq(h).all()
            and pred["model"].eq("XGBoost").all()
            and pred["split"].eq("test").all(),
            f"Horizon {h}: sai metadata du bao",
        )

        pred = pred.sort_values("observation_date").reset_index(drop=True)
        source = source.sort_values("observation_date").reset_index(drop=True)

        check(
            pred["observation_date"].equals(source["observation_date"]),
            f"Horizon {h}: ngay quan sat khong khop TV3",
        )
        check(
            pred["target_date"].equals(source[f"target_date_{h}"]),
            f"Horizon {h}: ngay muc tieu khong khop TV3",
        )
        np.testing.assert_allclose(
            pred["actual"], source[f"target_{h}"],
            rtol=0, atol=1e-9,
        )
        check(
            np.isfinite(pred[["actual", "predicted"]].to_numpy()).all(),
            f"Horizon {h}: co gia tri khong huu han",
        )

        artifact = joblib.load(ROOT / f"models/xgb_horizon_{h}.joblib")
        features = artifact["features"]
        model = artifact["model"]

        expected_features = [
            c for c in df.columns
            if c != "observation_date"
            and not c.startswith(("target_", "split_"))
        ]
        check(
            features == expected_features and len(features) == 29,
            f"Horizon {h}: sai danh sach dac trung",
        )
        check(
            artifact["horizon"] == h
            and artifact["stage"] == "final_trainval",
            f"Horizon {h}: sai artifact mo hinh",
        )

        recalculated = model.predict(source[features])
        np.testing.assert_allclose(
            pred["predicted"], recalculated,
            rtol=1e-6, atol=1e-6,
        )

        tuning = json.loads(
            (ROOT / f"results/tuning/xgb_tuning_summary_h{h}.json")
            .read_text(encoding="utf-8")
        )
        check(
            model.get_booster().num_boosted_rounds()
            == tuning["best_trial"]["best_n_estimators"],
            f"Horizon {h}: sai so vong boosting",
        )

        # Kiểm tra lại mốc nhãn dùng huấn luyện cuối.
        trainval = df[df[f"split_{h}"].isin(["train", "val"])].copy()
        cutoff = pd.to_datetime(source["observation_date"]).min()
        target_dates = pd.to_datetime(trainval[f"target_date_{h}"])
        safe = trainval[target_dates <= cutoff]

        check(
            artifact["first_test_origin"] == cutoff.strftime("%Y-%m-%d"),
            f"Horizon {h}: sai moc bat dau Test",
        )
        check(
            artifact["training_target_date_max"]
            == safe[f"target_date_{h}"].max(),
            f"Horizon {h}: sai moc nhan huan luyen",
        )

        importance = pd.read_csv(
            ROOT / f"results/xgb_importance_h{h}.csv"
        )
        check(
            len(importance) == 29
            and importance["feature"].is_unique
            and set(importance["feature"]) == set(features),
            f"Horizon {h}: sai danh sach importance",
        )
        recorded = importance.set_index("feature").loc[
            features, "importance_gain"
        ]
        np.testing.assert_allclose(
            recorded, model.feature_importances_,
            rtol=1e-5, atol=1e-8,
        )

        parts.append(pred)
        print(
            f"Horizon {h}: DAT | 442 mau Test"
            " | mo hinh, du bao va importance khop nhau"
        )

    combined = pd.read_csv(ROOT / "results/xgb_predictions_test.csv")
    expected = pd.concat(parts, ignore_index=True)

    keys = ["horizon", "observation_date", "target_date"]
    combined = combined.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        combined,
        expected,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
    print("File tong hop: DAT | 884 dong")
    print("\nKIEM TRA BO BAN GIAO: DAT")


if __name__ == "__main__":
    main()