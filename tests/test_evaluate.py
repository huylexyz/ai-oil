"""Acceptance and edge-case checks for Backtesting metrics."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest_data import load_backtest
from src.config import PROCESSED_DATA_DIR
from src.evaluate import METRIC_COLUMNS, calculate_metrics


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backtest = load_backtest(split="test")
        cls.metrics = calculate_metrics(cls.backtest).set_index(["model", "horizon"])

    def test_accepted_test_errors(self):
        expected = {
            ("Naive", 1): (1.9431, 3.0674),
            ("Naive", 5): (4.4405, 6.6461),
            ("Ridge", 1): (1.9124, 3.0645),
            ("Ridge", 5): (4.4637, 6.8299),
            ("Random Forest", 1): (1.9160, 2.9161),
            ("Random Forest", 5): (4.7220, 6.4765),
            ("XGBoost", 1): (1.9476, 3.0278),
            ("XGBoost", 5): (4.5922, 6.7681),
        }
        self.assertEqual(len(self.metrics), 8)
        for key, (mae, rmse) in expected.items():
            with self.subTest(key=key):
                actual = self.metrics.loc[key]
                self.assertEqual(actual["n_samples"], 442)
                self.assertEqual(actual["split"], "test")
                self.assertAlmostEqual(actual["mae"], mae, places=4)
                self.assertAlmostEqual(actual["rmse"], rmse, places=4)
                source = self.backtest.loc[
                    self.backtest["model"].eq(key[0]) & self.backtest["horizon"].eq(key[1])
                ]
                error = source["actual"].to_numpy() - source["predicted"].to_numpy()
                self.assertAlmostEqual(actual["mae"], np.abs(error).sum() / 442, places=12)
                self.assertAlmostEqual(actual["rmse"], np.sqrt(np.dot(error, error) / 442), places=12)

    def test_direction_accuracy_independently_from_tv2_prices(self):
        prices = pd.read_csv(PROCESSED_DATA_DIR / "brent_clean.csv")
        origin_price = dict(zip(prices["observation_date"], prices["DCOILBRENTEU"]))

        def direction(change):
            return int(change > 0) - int(change < 0)

        for (name, horizon), rows in self.backtest.groupby(["model", "horizon"]):
            reported = self.metrics.loc[(name, horizon), "directional_accuracy"]
            if name == "Naive":
                self.assertTrue(pd.isna(reported))
                continue
            hits = sum(
                direction(row.actual - origin_price[row.observation_date.strftime("%Y-%m-%d")])
                == direction(row.predicted - origin_price[row.observation_date.strftime("%Y-%m-%d")])
                for row in rows.itertuples(index=False)
            )
            self.assertAlmostEqual(reported, hits * 100 / len(rows), places=12)

    def test_zero_change_is_its_own_direction_and_naive_is_na(self):
        first = pd.read_csv(PROCESSED_DATA_DIR / "brent_clean.csv").iloc[0]
        price = float(first["DCOILBRENTEU"])
        rows = pd.DataFrame({
            "observation_date": [first["observation_date"]] * 4,
            "target_date": ["2015-01-05"] * 4,
            "horizon": [1] * 4,
            "actual": [price + 1, price - 1, price, price + 1],
            "predicted": [price + 2, price + 1, price, price],
            "model": ["Ridge", "Ridge", "Ridge", "Naive"],
            "split": ["test"] * 4,
        })
        result = calculate_metrics(rows).set_index("model")
        self.assertAlmostEqual(result.loc["Ridge", "directional_accuracy"], 200 / 3)
        self.assertTrue(pd.isna(result.loc["Naive", "directional_accuracy"]))

    def test_filtered_model_horizon_and_dates(self):
        rows = self.backtest.loc[
            self.backtest["model"].eq("Ridge")
            & self.backtest["horizon"].eq(5)
            & self.backtest["target_date"].between("2025-01-01", "2025-01-31")
        ]
        result = calculate_metrics(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "model"], "Ridge")
        self.assertEqual(result.loc[0, "horizon"], 5)
        self.assertEqual(result.loc[0, "n_samples"], len(rows))
        self.assertGreater(len(rows), 0)
        self.assertLess(len(rows), 442)

    def test_empty_and_missing_columns(self):
        empty = calculate_metrics(self.backtest.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertEqual(empty.columns.tolist(), list(METRIC_COLUMNS))
        with self.assertRaisesRegex(ValueError, "thiếu cột"):
            calculate_metrics(self.backtest.drop(columns="actual"))

    def test_invalid_values_and_dates(self):
        row = self.backtest.iloc[[0]].copy()
        for column, value in (
            ("actual", np.nan),
            ("predicted", np.inf),
            ("horizon", 1.5),
            ("model", ""),
        ):
            with self.subTest(column=column):
                bad = row.copy()
                bad[column] = value
                with self.assertRaises(ValueError):
                    calculate_metrics(bad)
        bad = row.copy()
        bad["target_date"] = bad["observation_date"]
        with self.assertRaisesRegex(ValueError, "target_date"):
            calculate_metrics(bad)
        bad = row.copy()
        bad["observation_date"] = pd.Timestamp("1900-01-01")
        with self.assertRaisesRegex(ValueError, "Không tìm thấy giá TV2"):
            calculate_metrics(bad)

    def test_tv2_column_names_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_columns.csv"
            path.write_text("date,price\n2025-01-01,50\n", encoding="utf-8")
            with patch("src.evaluate.BASE_PRICE_FILE", path):
                with self.assertRaisesRegex(ValueError, "Dữ liệu TV2 thiếu cột"):
                    calculate_metrics(self.backtest.iloc[[0]])


if __name__ == "__main__":
    unittest.main()
