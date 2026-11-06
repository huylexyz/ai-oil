"""Acceptance checks for TV6 XGBoost forecasts at horizons 2, 3 and 4."""

import json
import unittest

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.features import FEATURE_COLUMNS
from tv6.horizons_2_4 import CANDIDATES, HORIZONS, PREDICTION_COLUMNS, ROOT


class TV6NewHorizonsTests(unittest.TestCase):
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
            h: pd.read_csv(ROOT / "results" / f"xgb_predictions_test_h{h}.csv")
            for h in HORIZONS
        }
        cls.artifacts = {
            h: joblib.load(ROOT / "models" / f"xgb_horizon_{h}.joblib")
            for h in HORIZONS
        }
        cls.tuning = {
            h: pd.read_csv(ROOT / "results" / "tuning" / f"xgb_tuning_h{h}.csv")
            for h in HORIZONS
        }
        cls.tuning_summary = {
            h: json.loads((ROOT / "results" / "tuning" / f"xgb_tuning_summary_h{h}.json").read_text(encoding="utf-8"))
            for h in HORIZONS
        }
        cls.summary = {
            h: json.loads((ROOT / "results" / f"xgb_summary_h{h}.json").read_text(encoding="utf-8"))["horizons"][0]
            for h in HORIZONS
        }

    def test_test_csv_matches_tv3_tv4_tv5(self):
        for h in HORIZONS:
            with self.subTest(horizon=h):
                test = self.source[h].loc[self.source[h][f"split_{h}"].eq("test")]
                frame = self.predictions[h]
                tv4 = pd.read_csv(ROOT.parent / "tv4" / "predictions" / f"ridge_horizon_{h}.csv")
                tv5 = pd.read_csv(ROOT.parent / "tv5" / "results" / f"rf_predictions_test_t{h}.csv")
                self.assertEqual(frame.columns.tolist(), list(PREDICTION_COLUMNS))
                self.assertEqual(len(frame), len(test))
                self.assertEqual(len(frame), 442)
                self.assertFalse(frame.isna().any().any())
                self.assertFalse(frame["target_date"].duplicated().any())
                self.assertTrue(frame["horizon"].eq(h).all())
                self.assertTrue(frame["model"].eq("XGBoost").all())
                self.assertTrue(frame["split"].eq("test").all())
                self.assertEqual(frame["observation_date"].tolist(), test["observation_date"].dt.strftime("%Y-%m-%d").tolist())
                self.assertEqual(frame["target_date"].tolist(), test[f"target_date_{h}"].dt.strftime("%Y-%m-%d").tolist())
                self.assertEqual(frame["target_date"].tolist(), tv4["target_date"].tolist())
                self.assertEqual(frame["target_date"].tolist(), tv5["target_date"].tolist())
                np.testing.assert_allclose(frame["actual"], test[f"target_{h}"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(frame["actual"], tv4["actual"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(frame["actual"], tv5["actual"], rtol=0, atol=1e-12)

    def test_final_artifact_reproduces_csv_with_29_features(self):
        features = list(FEATURE_COLUMNS)
        self.assertEqual(len(features), 29)
        for h in HORIZONS:
            with self.subTest(horizon=h):
                artifact = self.artifacts[h]
                model = artifact["model"]
                test = self.source[h].loc[self.source[h][f"split_{h}"].eq("test")]
                self.assertEqual(artifact["horizon"], h)
                self.assertEqual(artifact["stage"], "final_trainval")
                self.assertEqual(artifact["features"], features)
                self.assertEqual(model.feature_names_in_.tolist(), features)
                self.assertEqual(model.n_features_in_, 29)
                self.assertEqual(model.get_params()["random_state"], 42)
                self.assertIsNone(model.get_params()["early_stopping_rounds"])
                self.assertEqual(model.get_params()["n_estimators"], self.summary[h]["n_estimators"])
                for name in ("max_depth", "learning_rate", "min_child_weight", "n_estimators", "random_state"):
                    self.assertEqual(artifact["params"][name], model.get_params()[name])
                recreated = model.predict(test[features])
                np.testing.assert_allclose(recreated, self.predictions[h]["predicted"], rtol=1e-7, atol=1e-7)

    def test_tuning_uses_18_validation_trials(self):
        self.assertEqual(len(CANDIDATES), 18)
        for h in HORIZONS:
            with self.subTest(horizon=h):
                tuning = self.tuning[h]
                summary = self.tuning_summary[h]
                self.assertEqual(len(tuning), 18)
                self.assertTrue(tuning["horizon"].eq(h).all())
                self.assertEqual(
                    set(zip(tuning["max_depth"], tuning["learning_rate"], tuning["min_child_weight"])),
                    set(CANDIDATES),
                )
                self.assertNotIn("test_MAE", tuning.columns)
                self.assertNotIn("test_RMSE", tuning.columns)
                best = tuning.sort_values(["validation_RMSE", "trial"]).iloc[0]
                self.assertEqual(summary["selection_metric"], "validation_RMSE")
                self.assertEqual(summary["best_trial"]["trial"], int(best["trial"]))
                self.assertAlmostEqual(summary["best_trial"]["validation_RMSE"], best["validation_RMSE"], places=10)
                selected = joblib.load(ROOT / "results" / "tuning" / f"xgb_selected_h{h}.joblib")
                validation = self.source[h].loc[self.source[h][f"split_{h}"].eq("val")]
                saved = pd.read_csv(ROOT / "results" / "tuning" / f"xgb_selected_validation_h{h}.csv")
                self.assertEqual(selected["features"], list(FEATURE_COLUMNS))
                self.assertEqual(selected["stage"], "validation_tuning")
                self.assertEqual(len(saved), len(validation))
                self.assertEqual(len(saved), 505)
                np.testing.assert_allclose(
                    selected["model"].predict(validation[list(FEATURE_COLUMNS)]),
                    saved["predicted"], rtol=1e-7, atol=1e-7,
                )

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
                summary = self.summary[h]
                self.assertEqual(int(train_purge.sum()), h - 1)
                self.assertEqual(int(val_purge.sum()), h - 1)
                self.assertEqual(summary["purged_train_boundary_rows"], h - 1)
                self.assertEqual(summary["purged_validation_boundary_rows"], h - 1)
                self.assertEqual(summary["train_rows_used_for_tuning"], len(train) - h + 1)
                self.assertEqual(summary["validation_rows_used_for_final_fit"], len(val) - h + 1)
                self.assertEqual(summary["final_training_rows"], len(train) + len(val) - h + 1)
                self.assertEqual(
                    summary["purged_train_observation_dates"],
                    train.loc[train_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
                )
                self.assertEqual(
                    summary["purged_validation_observation_dates"],
                    val.loc[val_purge, "observation_date"].dt.strftime("%Y-%m-%d").tolist(),
                )
                self.assertTrue(train.loc[~train_purge, target_date].le(first_val).all())
                self.assertTrue(val.loc[~val_purge, target_date].le(first_test).all())
                self.assertLessEqual(summary["training_target_date_max"], first_test.strftime("%Y-%m-%d"))

    def test_test_metrics_match_summary(self):
        expected = {
            2: (2.760181, 4.207893),
            3: (3.593453, 5.283781),
            4: (4.030534, 6.078837),
        }
        for h in HORIZONS:
            with self.subTest(horizon=h):
                frame = self.predictions[h]
                error = frame["predicted"].to_numpy() - frame["actual"].to_numpy()
                mae = float(np.abs(error).mean())
                rmse = float(np.sqrt(np.square(error).mean()))
                self.assertAlmostEqual(mae, expected[h][0], places=5)
                self.assertAlmostEqual(rmse, expected[h][1], places=5)
                np.testing.assert_allclose(self.summary[h]["test_MAE"], mae, rtol=0, atol=5e-7)
                np.testing.assert_allclose(self.summary[h]["test_RMSE"], rmse, rtol=0, atol=5e-7)

    def test_new_summaries_use_strict_json(self):
        for h in HORIZONS:
            for path in (
                ROOT / "results" / f"xgb_summary_h{h}.json",
                ROOT / "results" / "tuning" / f"xgb_tuning_summary_h{h}.json",
            ):
                with self.subTest(path=path):
                    json.loads(
                        path.read_text(encoding="utf-8"),
                        parse_constant=lambda token: self.fail(f"Nonstandard JSON constant: {token}"),
                    )


if __name__ == "__main__":
    unittest.main()
