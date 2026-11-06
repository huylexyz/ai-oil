"""Acceptance tests for TV5 Random Forest horizons 2, 3 and 4."""

import json
import unittest

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.features import FEATURE_COLUMNS
from tv5.horizons_2_4 import HORIZONS, PREDICTION_COLUMNS, ROOT
from tv5.train_rf import PARAM_GRID, make_features, summary_params


class TV5NewHorizonsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = {
            h: pd.read_csv(
                PROCESSED_DATA_DIR / f"brent_features_h{h}.csv",
                parse_dates=["observation_date", f"target_date_{h}"],
            )
            for h in HORIZONS
        }
        cls.predictions = {
            h: pd.read_csv(ROOT / "results" / f"rf_predictions_test_t{h}.csv")
            for h in HORIZONS
        }
        cls.models = {
            h: joblib.load(ROOT / "models" / f"rf_model_t{h}.joblib")
            for h in HORIZONS
        }
        cls.metadata = {
            h: json.loads((ROOT / "models" / f"rf_model_t{h}_metadata.json").read_text(encoding="utf-8"))
            for h in HORIZONS
        }
        cls.summary = {
            h: json.loads((ROOT / "results" / f"rf_summary_t{h}.json").read_text(encoding="utf-8"))
            for h in HORIZONS
        }

    def test_official_test_rows_and_tv4_alignment(self):
        for h in HORIZONS:
            with self.subTest(horizon=h):
                source_test = self.source[h].loc[self.source[h][f"split_{h}"].eq("test")]
                prediction = self.predictions[h]
                tv4 = pd.read_csv(ROOT.parent / "tv4" / "predictions" / f"ridge_horizon_{h}.csv")
                self.assertEqual(prediction.columns.tolist(), list(PREDICTION_COLUMNS))
                self.assertEqual(len(prediction), 442)
                self.assertEqual(len(source_test), 442)
                self.assertFalse(prediction.isna().any().any())
                self.assertFalse(prediction["target_date"].duplicated().any())
                self.assertTrue(prediction["horizon"].eq(h).all())
                self.assertTrue(prediction["model"].eq("RandomForest").all())
                self.assertTrue(prediction["split"].eq("test").all())
                self.assertEqual(
                    prediction["observation_date"].tolist(),
                    source_test["observation_date"].dt.strftime("%Y-%m-%d").tolist(),
                )
                self.assertEqual(
                    prediction["target_date"].tolist(),
                    source_test[f"target_date_{h}"].dt.strftime("%Y-%m-%d").tolist(),
                )
                self.assertEqual(prediction["target_date"].tolist(), tv4["target_date"].tolist())
                np.testing.assert_allclose(prediction["actual"], source_test[f"target_{h}"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(prediction["actual"], tv4["actual"], rtol=0, atol=1e-12)
                self.assertTrue(
                    (pd.to_datetime(prediction["observation_date"]) < pd.to_datetime(prediction["target_date"])).all()
                )

    def test_artifact_recreates_predictions_with_29_approved_features(self):
        features = list(FEATURE_COLUMNS)
        self.assertEqual(len(features), 29)
        for h in HORIZONS:
            with self.subTest(horizon=h):
                metadata = self.metadata[h]
                model = self.models[h]
                source_test = self.source[h].loc[self.source[h][f"split_{h}"].eq("test")]
                self.assertEqual(make_features(self.source[h]).columns.tolist(), features)
                self.assertEqual(metadata["features"], features)
                self.assertEqual(metadata["n_features"], 29)
                self.assertEqual(model.n_features_in_, 29)
                self.assertEqual(model.feature_names_in_.tolist(), features)
                self.assertEqual(model.random_state, 42)
                self.assertEqual(model.get_params()["n_estimators"], metadata["best_params"]["n_estimators"])
                self.assertEqual(model.get_params()["max_depth"], metadata["best_params"]["max_depth"])
                self.assertEqual(model.get_params()["min_samples_leaf"], metadata["best_params"]["min_samples_leaf"])
                self.assertEqual(model.get_params()["max_features"], metadata["best_params"]["max_features"])
                recreated = model.predict(source_test[features])
                np.testing.assert_allclose(recreated, self.predictions[h]["predicted"], rtol=0, atol=1e-10)

    def test_purge_uses_actual_target_dates(self):
        for h in HORIZONS:
            with self.subTest(horizon=h):
                source = self.source[h]
                train = source.loc[source[f"split_{h}"].eq("train")]
                val = source.loc[source[f"split_{h}"].eq("val")]
                test = source.loc[source[f"split_{h}"].eq("test")]
                target_date = f"target_date_{h}"
                first_val = val["observation_date"].min()
                first_test = test["observation_date"].min()
                train_purge = train[target_date].gt(first_val)
                val_purge = val[target_date].gt(first_test)
                metadata = self.metadata[h]
                purge = metadata["purged_boundary_samples"]
                self.assertEqual(purge["train_val"], int(train_purge.sum()))
                self.assertEqual(purge["train_val"], h - 1)
                self.assertEqual(purge["val_test"], int(val_purge.sum()))
                self.assertEqual(purge["val_test"], h - 1)
                self.assertEqual(
                    purge["train_val_observation_dates"],
                    train.loc[train_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
                )
                self.assertEqual(
                    purge["val_test_observation_dates"],
                    val.loc[val_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
                )
                self.assertEqual(metadata["train_rows_used_for_tuning"], len(train) - h + 1)
                self.assertEqual(metadata["trainval_rows_used_for_final_fit"], len(train) + len(val) - h + 1)
                self.assertTrue(train.loc[~train_purge, target_date].le(first_val).all())
                self.assertTrue(val.loc[~val_purge, target_date].le(first_test).all())
                self.assertLessEqual(metadata["trainval_target_date_range"][1], first_test.strftime("%Y-%m-%d"))

    def test_hyperparameters_selected_by_validation_rmse(self):
        for h in HORIZONS:
            with self.subTest(horizon=h):
                tuning = pd.read_csv(ROOT / "results" / f"rf_tuning_h{h}.csv")
                metadata = self.metadata[h]
                self.assertEqual(len(tuning), len(PARAM_GRID))
                self.assertEqual(len(tuning), 6)
                self.assertTrue(tuning["horizon"].eq(h).all())
                self.assertNotIn("test_MAE", tuning.columns)
                self.assertNotIn("test_RMSE", tuning.columns)
                best = tuning.loc[tuning["validation_RMSE"].idxmin()].to_dict()
                self.assertEqual(metadata["selection_metric"], "validation_RMSE")
                self.assertEqual(metadata["best_params"], summary_params(best))
                self.assertAlmostEqual(metadata["best_validation_MAE"], best["validation_MAE"], places=10)
                self.assertAlmostEqual(metadata["best_validation_RMSE"], best["validation_RMSE"], places=10)

    def test_test_metrics_match_saved_results(self):
        for h in HORIZONS:
            with self.subTest(horizon=h):
                prediction = self.predictions[h]
                error = prediction["predicted"].to_numpy() - prediction["actual"].to_numpy()
                mae = np.abs(error).mean()
                rmse = np.sqrt(np.square(error).mean())
                self.assertAlmostEqual(self.summary[h]["test_MAE"], mae, places=10)
                self.assertAlmostEqual(self.summary[h]["test_RMSE"], rmse, places=10)
                self.assertAlmostEqual(self.metadata[h]["test_MAE"], mae, places=10)
                self.assertAlmostEqual(self.metadata[h]["test_RMSE"], rmse, places=10)


if __name__ == "__main__":
    unittest.main()
