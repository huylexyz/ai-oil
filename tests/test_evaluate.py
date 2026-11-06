"""Acceptance and edge-case checks for Backtesting metrics."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest_data import SOURCES, load_backtest
from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.evaluate import METRIC_COLUMNS, calculate_metrics


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backtest = load_backtest(split="test")
        cls.metrics_df = calculate_metrics(cls.backtest)
        cls.metrics = cls.metrics_df.set_index(["model", "horizon"])

    def test_metric_schema_and_all_twenty_groups(self):
        self.assertEqual(self.metrics_df.columns.tolist(), list(METRIC_COLUMNS))
        self.assertEqual(len(self.metrics_df), 20)
        self.assertFalse(self.metrics.index.has_duplicates)
        self.assertEqual(set(self.metrics.index.get_level_values("model")),
                         {"Naive", "Ridge", "Random Forest", "XGBoost"})
        self.assertEqual(set(self.metrics.index.get_level_values("horizon")),
                         {1, 2, 3, 4, 5})
        self.assertTrue(self.metrics_df["n_samples"].eq(442).all())
        self.assertTrue(self.metrics_df["split"].eq("test").all())
        self.assertTrue(self.metrics_df[["mae", "rmse"]].ge(0).all().all())

    def test_accepted_test_errors(self):
        expected = {
            ("Naive", 1): (1.9431, 3.0674),
            ("Naive", 2): (2.7798, 4.2703),
            ("Naive", 3): (3.4797, 5.2305),
            ("Naive", 4): (4.0442, 5.9696),
            ("Naive", 5): (4.4405, 6.6461),
            ("Ridge", 1): (1.9124, 3.0645),
            ("Ridge", 2): (2.7785, 4.2488),
            ("Ridge", 3): (3.5225, 5.2637),
            ("Ridge", 4): (4.0668, 6.1800),
            ("Ridge", 5): (4.4637, 6.8299),
            ("Random Forest", 1): (1.9160, 2.9161),
            ("Random Forest", 2): (2.805443, 4.143301),
            ("Random Forest", 3): (3.561162, 5.000933),
            ("Random Forest", 4): (4.175802, 5.712405),
            ("Random Forest", 5): (4.7220, 6.4765),
            ("XGBoost", 1): (1.9476, 3.0278),
            ("XGBoost", 2): (2.760181, 4.207893),
            ("XGBoost", 3): (3.593453, 5.283781),
            ("XGBoost", 4): (4.030534, 6.078837),
            ("XGBoost", 5): (4.5922, 6.7681),
        }
        self.assertEqual(len(self.metrics), 20)
        for key, (mae, rmse) in expected.items():
            with self.subTest(key=key):
                actual = self.metrics.loc[key]
                self.assertEqual(actual["n_samples"], 442)
                self.assertEqual(actual["split"], "test")
                tolerance = 1e-6 if key[0] in ("Random Forest", "XGBoost") and key[1] in (2, 3, 4) else 5e-5
                self.assertAlmostEqual(actual["mae"], mae, delta=tolerance)
                self.assertAlmostEqual(actual["rmse"], rmse, delta=tolerance)
                source = self.backtest.loc[
                    self.backtest["model"].eq(key[0]) & self.backtest["horizon"].eq(key[1])
                ]
                error = source["actual"].to_numpy() - source["predicted"].to_numpy()
                self.assertAlmostEqual(actual["mae"], np.abs(error).sum() / 442, places=12)
                self.assertAlmostEqual(actual["rmse"], np.sqrt(np.dot(error, error) / 442), places=12)

    def test_mae_rmse_against_official_csvs(self):
        for name, horizon, relative_path, _ in SOURCES:
            with self.subTest(model=name, horizon=horizon):
                official = pd.read_csv(PROJECT_ROOT / relative_path)
                official = official.loc[official["split"].eq("test")]
                error = official["actual"].to_numpy() - official["predicted"].to_numpy()
                self.assertEqual(len(error), 442)
                reported = self.metrics.loc[(name, horizon)]
                self.assertAlmostEqual(reported["mae"], np.abs(error).mean(), places=12)
                self.assertAlmostEqual(reported["rmse"], np.sqrt(np.mean(error ** 2)), places=12)

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
            self.assertGreaterEqual(reported, 0)
            self.assertLessEqual(reported, 100)
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

        model_only = calculate_metrics(self.backtest.loc[self.backtest["model"].eq("Ridge")])
        self.assertEqual(set(model_only["horizon"]), {1, 2, 3, 4, 5})
        horizon_only = calculate_metrics(self.backtest.loc[self.backtest["horizon"].eq(3)])
        self.assertEqual(set(horizon_only["model"]), {"Naive", "Ridge", "Random Forest", "XGBoost"})
        january = self.backtest.loc[self.backtest["target_date"].between("2025-01-01", "2025-01-31")]
        january_metrics = calculate_metrics(january)
        self.assertEqual(len(january_metrics), 20)
        for row in january_metrics.itertuples(index=False):
            count = len(january.loc[january["model"].eq(row.model) & january["horizon"].eq(row.horizon)])
            self.assertEqual(row.n_samples, count)
            self.assertGreater(count, 0)
            self.assertLess(count, 442)

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
