"""Checks for forecasting from approved, unchanged model artifacts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.features import FEATURE_COLUMNS, build_features
from src.predict import CLEAN_DATA_FILE, MODEL_FILES, RF_METADATA_FILES, predict_latest


class PredictLatestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = pd.read_csv(CLEAN_DATA_FILE)

    def test_all_eight_latest_predictions_and_formulas(self):
        latest = self.clean.iloc[-1]
        for model in ("Naive", "Ridge", "Random Forest", "XGBoost"):
            for horizon in (1, 5):
                with self.subTest(model=model, horizon=horizon):
                    result = predict_latest(model, horizon)
                    self.assertEqual(list(result), [
                        "model", "horizon", "observation_date", "last_price",
                        "predicted_price", "absolute_change", "percent_change",
                    ])
                    self.assertEqual(result["model"], model)
                    self.assertEqual(result["horizon"], horizon)
                    self.assertEqual(result["observation_date"], latest["observation_date"])
                    self.assertEqual(result["last_price"], latest["DCOILBRENTEU"])
                    self.assertTrue(np.isfinite(result["predicted_price"]))
                    self.assertAlmostEqual(
                        result["absolute_change"], result["predicted_price"] - result["last_price"]
                    )
                    self.assertAlmostEqual(
                        result["percent_change"], result["absolute_change"] / result["last_price"] * 100
                    )
                    if model == "Naive":
                        self.assertEqual(result["predicted_price"], result["last_price"])

    def test_two_official_test_observations_for_all_eight_combinations(self):
        source = pd.read_csv(PROCESSED_DATA_DIR / "brent_features.csv")
        dates = source.loc[source["split_1"].eq("test"), "observation_date"].iloc[[0, 200]]
        csv_paths = {
            ("Naive", h): PROJECT_ROOT / f"tv4/predictions/naive_horizon_{h}.csv" for h in (1, 5)
        } | {
            ("Ridge", h): PROJECT_ROOT / f"tv4/predictions/ridge_horizon_{h}.csv" for h in (1, 5)
        } | {
            ("Random Forest", h): PROJECT_ROOT / f"tv5/results/rf_predictions_test_t{h}.csv" for h in (1, 5)
        } | {
            ("XGBoost", h): PROJECT_ROOT / f"tv6/results/xgb_predictions_test_h{h}.csv" for h in (1, 5)
        }
        with tempfile.TemporaryDirectory() as directory:
            clean_path = Path(directory) / "brent_clean.csv"
            for date in dates:
                self.clean.loc[self.clean["observation_date"].le(date)].to_csv(clean_path, index=False)
                with patch("src.predict.CLEAN_DATA_FILE", clean_path):
                    for (model, horizon), csv_path in csv_paths.items():
                        with self.subTest(date=date, model=model, horizon=horizon):
                            expected = pd.read_csv(csv_path)
                            expected = expected.loc[
                                expected["split"].eq("test") & expected["observation_date"].eq(date)
                            ]
                            self.assertEqual(len(expected), 1)
                            result = predict_latest(model, horizon)
                            self.assertEqual(result["observation_date"], date)
                            np.testing.assert_allclose(
                                result["predicted_price"], expected.iloc[0]["predicted"],
                                rtol=1e-6, atol=1e-6,
                            )

    def test_model_and_horizon_rejected(self):
        for model, horizon in (("Unknown", 1), ("Ridge", 2), ("Ridge", True), ("Ridge", "1")):
            with self.subTest(model=model, horizon=horizon):
                with self.assertRaises(ValueError):
                    predict_latest(model, horizon)

    def test_missing_and_invalid_clean_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with patch("src.predict.CLEAN_DATA_FILE", path):
                with self.assertRaises(FileNotFoundError):
                    predict_latest("Naive", 1)
                self.clean.iloc[:20].to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "Không đủ dữ liệu"):
                    predict_latest("Naive", 1)
                bad = self.clean.copy()
                bad.loc[bad.index[-1], "DCOILBRENTEU"] = 0
                bad.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "số dương"):
                    predict_latest("Naive", 1)

    def test_missing_and_wrong_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.joblib"
            with patch.dict(MODEL_FILES, {("Ridge", 1): missing}):
                with self.assertRaises(FileNotFoundError):
                    predict_latest("Ridge", 1)
            wrong_horizon = MODEL_FILES[("Ridge", 5)]
            with patch.dict(MODEL_FILES, {("Ridge", 1): wrong_horizon}):
                with self.assertRaisesRegex(ValueError, "horizon"):
                    predict_latest("Ridge", 1)
            metadata = json.loads(RF_METADATA_FILES[1].read_text(encoding="utf-8"))
            metadata["features"] = list(reversed(metadata["features"]))
            path = Path(directory) / "wrong_order.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with patch.dict(RF_METADATA_FILES, {1: path}):
                with self.assertRaisesRegex(ValueError, "thứ tự"):
                    predict_latest("Random Forest", 1)

    def test_missing_and_reordered_tv3_features(self):
        generated = build_features(self.clean)
        with patch("src.predict.build_features", return_value=generated.drop(columns=FEATURE_COLUMNS[0])):
            with self.assertRaisesRegex(ValueError, "đặc trưng"):
                predict_latest("Naive", 1)
        columns = list(generated.columns)
        a, b = columns.index(FEATURE_COLUMNS[0]), columns.index(FEATURE_COLUMNS[1])
        columns[a], columns[b] = columns[b], columns[a]
        with patch("src.predict.build_features", return_value=generated[columns]):
            with self.assertRaisesRegex(ValueError, "thứ tự"):
                predict_latest("Naive", 1)

    def test_invalid_prediction_is_rejected(self):
        class BadModel:
            def predict(self, inputs):
                return [np.nan]

        with patch("src.predict._load_model", return_value=BadModel()):
            with self.assertRaisesRegex(ValueError, "Giá dự báo"):
                predict_latest("Ridge", 1)


if __name__ == "__main__":
    unittest.main()
