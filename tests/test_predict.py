"""Checks for forecasting from approved, unchanged model artifacts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.features import FEATURE_COLUMNS, build_features
from src.predict import (
    CLEAN_DATA_FILE, HORIZONS, MODEL_FILES, RF_METADATA_FILES,
    _latest_input, _load_model, predict_latest,
)


class PredictLatestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = pd.read_csv(CLEAN_DATA_FILE)

    def test_all_twenty_latest_predictions_and_formulas(self):
        latest = self.clean.iloc[-1]
        self.assertEqual(latest["observation_date"], "2026-09-30")
        self.assertEqual(latest["DCOILBRENTEU"], 115.91)
        for model in ("Naive", "Ridge", "Random Forest", "XGBoost"):
            for horizon in HORIZONS:
                with self.subTest(model=model, horizon=horizon):
                    result = predict_latest(model, horizon)
                    self.assertEqual(list(result), [
                        "model", "horizon", "observation_date", "last_price",
                        "predicted_price", "absolute_change", "percent_change",
                    ])
                    self.assertEqual(result["model"], model)
                    self.assertEqual(result["horizon"], horizon)
                    self.assertIsInstance(result["horizon"], int)
                    self.assertIsInstance(result["observation_date"], str)
                    self.assertEqual(result["observation_date"], latest["observation_date"])
                    self.assertEqual(result["last_price"], latest["DCOILBRENTEU"])
                    for key in ("last_price", "predicted_price", "absolute_change", "percent_change"):
                        self.assertIsInstance(result[key], float)
                        self.assertTrue(np.isfinite(result[key]))
                    self.assertTrue(np.isfinite(result["predicted_price"]))
                    self.assertAlmostEqual(
                        result["absolute_change"], result["predicted_price"] - result["last_price"]
                    )
                    self.assertAlmostEqual(
                        result["percent_change"], result["absolute_change"] / result["last_price"] * 100
                    )
                    if model == "Naive":
                        self.assertEqual(result["predicted_price"], result["last_price"])

    def test_accepted_horizon_one_and_five_predictions_unchanged(self):
        expected = {
            ("Naive", 1): 115.91,
            ("Naive", 5): 115.91,
            ("Ridge", 1): 116.31612562184813,
            ("Ridge", 5): 115.52803924740522,
            ("Random Forest", 1): 116.016102526455,
            ("Random Forest", 5): 114.09971291666659,
            ("XGBoost", 1): 116.5431900024414,
            ("XGBoost", 5): 112.3205337524414,
        }
        for (model, horizon), price in expected.items():
            with self.subTest(model=model, horizon=horizon):
                result = predict_latest(model, horizon)
                np.testing.assert_allclose(result["predicted_price"], price, rtol=1e-10, atol=1e-8)

    def test_final_artifacts_and_latest_29_features(self):
        observation_date, last_price, latest = _latest_input()
        self.assertEqual(observation_date.strftime("%Y-%m-%d"), "2026-09-30")
        self.assertEqual(last_price, 115.91)
        self.assertEqual(latest.shape, (1, 29))
        self.assertEqual(latest.columns.tolist(), list(FEATURE_COLUMNS))
        self.assertFalse(any(name.startswith(("target_", "split_")) for name in latest.columns))
        for model in ("Ridge", "Random Forest", "XGBoost"):
            for horizon in HORIZONS:
                with self.subTest(model=model, horizon=horizon):
                    path = MODEL_FILES[(model, horizon)]
                    self.assertTrue(path.is_file(), path)
                    estimator = _load_model(model, horizon, latest.columns.tolist())
                    self.assertEqual(estimator.n_features_in_, 29)
                    self.assertEqual(list(estimator.feature_names_in_), list(FEATURE_COLUMNS))
                    if model == "XGBoost":
                        self.assertEqual(joblib.load(path)["stage"], "final_trainval")
                    if model == "Random Forest":
                        self.assertTrue(RF_METADATA_FILES[horizon].is_file())

    def test_two_official_test_observations_for_all_twenty_combinations(self):
        source = pd.read_csv(PROCESSED_DATA_DIR / "brent_features.csv")
        dates = source.loc[source["split_1"].eq("test"), "observation_date"].iloc[[0, 200]]
        csv_paths = {
            ("Naive", h): PROJECT_ROOT / f"tv4/predictions/naive_horizon_{h}.csv" for h in HORIZONS
        } | {
            ("Ridge", h): PROJECT_ROOT / f"tv4/predictions/ridge_horizon_{h}.csv" for h in HORIZONS
        } | {
            ("Random Forest", h): PROJECT_ROOT / f"tv5/results/rf_predictions_test_t{h}.csv" for h in HORIZONS
        } | {
            ("XGBoost", h): PROJECT_ROOT / f"tv6/results/xgb_predictions_test_h{h}.csv" for h in HORIZONS
        }
        self.assertEqual(len(csv_paths), 20)
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
        for model, horizon in (
            ("Unknown", 1), ("Ridge", 0), ("Ridge", 6),
            ("Ridge", True), ("Ridge", "1"), ("Ridge", 1.5),
        ):
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
            def __init__(self, value):
                self.value = value

            def predict(self, inputs):
                return [self.value]

        for value in (np.nan, np.inf, -1):
            with self.subTest(value=value):
                with patch("src.predict._load_model", return_value=BadModel(value)):
                    with self.assertRaisesRegex(ValueError, "Giá dự báo"):
                        predict_latest("Ridge", 2)

        class IncompatibleModel:
            def predict(self, inputs):
                raise ValueError("Sai số đặc trưng")

        with patch("src.predict._load_model", return_value=IncompatibleModel()):
            with self.assertRaisesRegex(RuntimeError, "không tương thích"):
                predict_latest("Ridge", 2)


if __name__ == "__main__":
    unittest.main()
