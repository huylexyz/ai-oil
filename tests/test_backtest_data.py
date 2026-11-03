"""Acceptance checks for the handed-off Backtesting forecasts."""

import unittest

import pandas as pd

from src.backtest_data import COLUMNS, load_backtest


class BacktestDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecasts = load_backtest()

    def test_schema_types_and_counts(self):
        frame = self.forecasts
        self.assertEqual(frame.columns.tolist(), list(COLUMNS))
        self.assertEqual(len(frame), 3536)
        self.assertEqual(set(frame["model"]), {"Naive", "Ridge", "Random Forest", "XGBoost"})
        self.assertEqual(set(frame["horizon"]), {1, 5})
        self.assertEqual(set(frame["split"]), {"test"})
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["observation_date"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["target_date"]))
        self.assertTrue(pd.api.types.is_integer_dtype(frame["horizon"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(frame["actual"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(frame["predicted"]))
        self.assertFalse(frame.isna().any().any())
        self.assertTrue(frame.groupby(["model", "horizon"]).size().eq(442).all())

    def test_keys_and_cross_model_alignment(self):
        frame = self.forecasts
        self.assertFalse(frame.duplicated(["model", "horizon", "split", "target_date"]).any())
        for horizon in (1, 5):
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

    def test_validation_reports_missing_official_forecasts(self):
        with self.assertRaisesRegex(ValueError, "thiếu Random Forest và XGBoost"):
            load_backtest(split="validation")


if __name__ == "__main__":
    unittest.main()
