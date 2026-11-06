"""Acceptance checks for the handed-off horizon 1–5 Backtesting forecasts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest_data import COLUMNS, SOURCES, load_backtest
from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT


class BacktestDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecasts = load_backtest()

    def test_schema_types_and_counts(self):
        frame = self.forecasts
        self.assertEqual(frame.columns.tolist(), list(COLUMNS))
        self.assertEqual(len(frame), 8840)
        self.assertEqual(set(frame["model"]), {"Naive", "Ridge", "Random Forest", "XGBoost"})
        self.assertEqual(set(frame["horizon"]), {1, 2, 3, 4, 5})
        self.assertEqual(set(frame["split"]), {"test"})
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["observation_date"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["target_date"]))
        self.assertTrue(pd.api.types.is_integer_dtype(frame["horizon"]))
        self.assertTrue(pd.api.types.is_float_dtype(frame["actual"]))
        self.assertTrue(pd.api.types.is_float_dtype(frame["predicted"]))
        self.assertFalse(frame.isna().any().any())
        self.assertTrue(np.isfinite(frame[["actual", "predicted"]].to_numpy()).all())
        counts = frame.groupby(["model", "horizon"]).size()
        self.assertEqual(len(counts), 20)
        self.assertTrue(counts.eq(442).all())

    def test_keys_and_cross_model_alignment(self):
        frame = self.forecasts
        self.assertFalse(frame.duplicated(["model", "horizon", "split", "target_date"]).any())
        first_horizon = None
        for horizon in (1, 2, 3, 4, 5):
            tables = {
                name: group.sort_values("target_date").reset_index(drop=True)
                for name, group in frame[frame["horizon"].eq(horizon)].groupby("model")
            }
            reference = tables["Naive"]
            for name in ("Ridge", "Random Forest", "XGBoost"):
                pd.testing.assert_frame_equal(
                    tables[name][["observation_date", "target_date", "actual"]],
                    reference[["observation_date", "target_date", "actual"]],
                    check_exact=False,
                    rtol=0,
                    atol=1e-9,
                )
            if first_horizon is None:
                first_horizon = reference
            else:
                self.assertEqual(reference["target_date"].tolist(), first_horizon["target_date"].tolist())
                np.testing.assert_allclose(reference["actual"], first_horizon["actual"], rtol=0, atol=1e-9)

    def test_forecasts_match_all_twenty_official_csvs(self):
        self.assertEqual(len(SOURCES), 20)
        for name, horizon, relative_path, raw_name in SOURCES:
            with self.subTest(model=name, horizon=horizon):
                path = PROJECT_ROOT / relative_path
                self.assertTrue(path.is_file(), path)
                official = pd.read_csv(path)
                official = official.loc[official["split"].eq("test")].reset_index(drop=True)
                loaded = self.forecasts.loc[
                    self.forecasts["model"].eq(name) & self.forecasts["horizon"].eq(horizon)
                ].reset_index(drop=True)
                self.assertEqual(len(official), len(loaded))
                self.assertTrue(official["model"].eq(raw_name).all())
                self.assertEqual(
                    loaded["observation_date"].dt.strftime("%Y-%m-%d").tolist(),
                    official["observation_date"].tolist(),
                )
                self.assertEqual(
                    loaded["target_date"].dt.strftime("%Y-%m-%d").tolist(),
                    official["target_date"].tolist(),
                )
                np.testing.assert_array_equal(loaded["actual"].to_numpy(), official["actual"].to_numpy())
                np.testing.assert_array_equal(loaded["predicted"].to_numpy(), official["predicted"].to_numpy())

    def test_dates_and_actual_match_tv3_handoffs(self):
        for horizon in (1, 2, 3, 4, 5):
            path = PROCESSED_DATA_DIR / (
                "brent_features.csv" if horizon in (1, 5) else f"brent_features_h{horizon}.csv"
            )
            official = pd.read_csv(path)
            official = official.loc[official[f"split_{horizon}"].eq("test")]
            official = official.sort_values(f"target_date_{horizon}").reset_index(drop=True)
            self.assertEqual(len(official), 442)
            for model in ("Naive", "Ridge", "Random Forest", "XGBoost"):
                with self.subTest(model=model, horizon=horizon):
                    loaded = self.forecasts.loc[
                        self.forecasts["model"].eq(model) & self.forecasts["horizon"].eq(horizon)
                    ].sort_values("target_date").reset_index(drop=True)
                    self.assertEqual(
                        loaded["observation_date"].dt.strftime("%Y-%m-%d").tolist(),
                        official["observation_date"].tolist(),
                    )
                    self.assertEqual(
                        loaded["target_date"].dt.strftime("%Y-%m-%d").tolist(),
                        official[f"target_date_{horizon}"].tolist(),
                    )
                    np.testing.assert_allclose(loaded["actual"], official[f"target_{horizon}"], rtol=0, atol=1e-9)

    def test_horizon_counts_valid_brent_sessions(self):
        clean = pd.read_csv(PROCESSED_DATA_DIR / "brent_clean.csv", parse_dates=["observation_date"])
        position = {date: index for index, date in enumerate(clean["observation_date"])}
        for horizon, rows in self.forecasts.groupby("horizon"):
            with self.subTest(horizon=horizon):
                offsets = [
                    position[target] - position[origin]
                    for origin, target in zip(rows["observation_date"], rows["target_date"])
                ]
                self.assertEqual(set(offsets), {horizon})

    def test_missing_csv_and_bad_schema_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.backtest_data.PROJECT_ROOT", root):
                with self.assertRaisesRegex(FileNotFoundError, "Thiếu CSV dự báo"):
                    load_backtest()
                first_path = root / SOURCES[0][2]
                first_path.parent.mkdir(parents=True)
                first_path.write_text(
                    "observation_date,target_date,horizon,actual,model,split\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "thiếu cột"):
                    load_backtest()
                first_path.write_text(
                    "target_date,observation_date,horizon,actual,predicted,model,split\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "sai schema"):
                    load_backtest()

    def test_validation_reports_missing_official_forecasts(self):
        with self.assertRaisesRegex(ValueError, "thiếu Random Forest và XGBoost"):
            load_backtest(split="validation")


if __name__ == "__main__":
    unittest.main()
