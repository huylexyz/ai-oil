"""Acceptance checks for TV4 Naive and Ridge Test forecasts at h2–h4."""

import unittest

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.features import FEATURE_COLUMNS
from tv4.horizons_2_4 import HORIZONS, PREDICTION_COLUMNS, ROOT, load_horizon_data, score_test
from tv4.ridge_regression import ALPHA_VALUES, FEATURES


class TV4NewHorizonsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = {h: load_horizon_data(h) for h in HORIZONS}
        cls.csv = {
            (model, h): pd.read_csv(ROOT / "predictions" / f"{name}_horizon_{h}.csv")
            for h in HORIZONS
            for model, name in (("Naive", "naive"), ("Ridge", "ridge"))
        }
        cls.artifacts = {
            h: joblib.load(ROOT / "models" / f"ridge_horizon_{h}.joblib")
            for h in HORIZONS
        }

    def test_prediction_schema_and_official_test_targets(self):
        official = pd.read_csv(PROCESSED_DATA_DIR / "brent_features.csv")
        reference = official.loc[official["split_1"].eq("test"), "target_date_1"].tolist()
        self.assertEqual(reference, official.loc[official["split_5"].eq("test"), "target_date_5"].tolist())
        for h in HORIZONS:
            test = self.input[h].loc[self.input[h][f"split_{h}"].eq("test")]
            for model in ("Naive", "Ridge"):
                with self.subTest(horizon=h, model=model):
                    frame = self.csv[(model, h)]
                    self.assertEqual(frame.columns.tolist(), list(PREDICTION_COLUMNS))
                    self.assertEqual(len(frame), 442)
                    self.assertFalse(frame.isna().any().any())
                    self.assertFalse(frame["target_date"].duplicated().any())
                    self.assertTrue(frame["horizon"].eq(h).all())
                    self.assertTrue(frame["model"].eq(model).all())
                    self.assertTrue(frame["split"].eq("test").all())
                    self.assertEqual(frame["observation_date"].tolist(), test["observation_date"].dt.strftime("%Y-%m-%d").tolist())
                    self.assertEqual(frame["target_date"].tolist(), reference)
                    np.testing.assert_allclose(frame["actual"], test[f"target_{h}"], rtol=0, atol=1e-12)
                    self.assertTrue((pd.to_datetime(frame["observation_date"]) < pd.to_datetime(frame["target_date"])).all())

    def test_naive_is_observation_price(self):
        for h in HORIZONS:
            with self.subTest(horizon=h):
                test = self.input[h].loc[self.input[h][f"split_{h}"].eq("test")]
                np.testing.assert_allclose(self.csv[("Naive", h)]["predicted"], test["price"], rtol=0, atol=1e-12)

    def test_ridge_artifact_reproduces_csv_with_approved_pipeline(self):
        self.assertEqual(FEATURES, list(FEATURE_COLUMNS))
        for h in HORIZONS:
            with self.subTest(horizon=h):
                artifact = self.artifacts[h]
                model = artifact["model"]
                self.assertEqual(artifact["horizon"], h)
                self.assertEqual(artifact["features"], FEATURES)
                self.assertIn(artifact["alpha"], ALPHA_VALUES)
                self.assertEqual(list(model.named_steps), ["imputer", "scaler", "ridge"])
                self.assertEqual(list(model.feature_names_in_), FEATURES)
                self.assertEqual(model.n_features_in_, 29)
                test = self.input[h].loc[self.input[h][f"split_{h}"].eq("test")]
                recreated = model.predict(test[FEATURES])
                np.testing.assert_allclose(recreated, self.csv[("Ridge", h)]["predicted"], rtol=0, atol=1e-10)

    def test_boundary_purge_uses_target_dates(self):
        for h in HORIZONS:
            with self.subTest(horizon=h):
                data = self.input[h]
                train = data.loc[data[f"split_{h}"].eq("train")]
                val = data.loc[data[f"split_{h}"].eq("val")]
                test = data.loc[data[f"split_{h}"].eq("test")]
                date_col = f"target_date_{h}"
                first_val = val["observation_date"].min()
                first_test = test["observation_date"].min()
                removed_train = train[date_col].gt(first_val)
                removed_val = val[date_col].gt(first_test)
                artifact = self.artifacts[h]
                self.assertEqual(int(removed_train.sum()), h - 1)
                self.assertEqual(int(removed_val.sum()), h - 1)
                self.assertEqual(artifact["excluded_train_boundary_rows_for_tuning"], int(removed_train.sum()))
                self.assertEqual(artifact["excluded_validation_boundary_rows_for_final_fit"], int(removed_val.sum()))
                self.assertEqual(artifact["tuning_train_rows"], len(train) - int(removed_train.sum()))
                self.assertEqual(artifact["validation_rows_used_for_final_fit"], len(val) - int(removed_val.sum()))
                self.assertEqual(artifact["training_rows"], len(train) + len(val) - int(removed_val.sum()))
                self.assertEqual(artifact["first_validation_observation_date"], first_val.strftime("%Y-%m-%d"))
                self.assertEqual(artifact["first_test_observation_date"], first_test.strftime("%Y-%m-%d"))
                self.assertTrue(train.loc[~removed_train, date_col].le(first_val).all())
                self.assertTrue(val.loc[~removed_val, date_col].le(first_test).all())

    def test_test_mae_rmse(self):
        expected = {
            ("Naive", 2): (2.7798, 4.2703),
            ("Ridge", 2): (2.7785, 4.2488),
            ("Naive", 3): (3.4797, 5.2305),
            ("Ridge", 3): (3.5225, 5.2637),
            ("Naive", 4): (4.0442, 5.9696),
            ("Ridge", 4): (4.0668, 6.1800),
        }
        for key, frame in self.csv.items():
            with self.subTest(model=key[0], horizon=key[1]):
                mae, rmse = score_test(frame)
                self.assertAlmostEqual(mae, expected[key][0], places=4)
                self.assertAlmostEqual(rmse, expected[key][1], places=4)


if __name__ == "__main__":
    unittest.main()
