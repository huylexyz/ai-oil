"""Frontend evaluation must preserve the official Test metrics and forecasts."""

import unittest

import numpy as np
import pandas as pd

from src.backtest_data import load_backtest
from src.core.config import MODELS_LIST, OFFICIAL_HORIZONS
from src.evaluate import METRIC_COLUMNS, calculate_metrics
from src.services.data_service import get_all_predictions
from src.services.forecast_service import compute_evaluation_metrics
from src.ui.charts import plot_bento_error_bars, plot_bento_test_comparison
from src.views.evaluation_view import _format_metrics_for_display


class EvaluationViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = load_backtest(split="test")
        cls.predictions, cls.has_predictions, cls.paths = get_all_predictions()
        cls.backend_metrics = calculate_metrics(cls.official)
        cls.service_metrics = compute_evaluation_metrics(cls.predictions, MODELS_LIST)

    def test_twenty_unrounded_groups_match_backend(self):
        self.assertTrue(self.has_predictions)
        self.assertEqual(len(self.predictions), 8840)
        self.assertEqual(len(self.paths), 20)
        self.assertEqual(len(self.service_metrics), 20)
        self.assertEqual(self.service_metrics.columns.tolist(), list(METRIC_COLUMNS))
        self.assertTrue(self.service_metrics["n_samples"].eq(442).all())
        for horizon in OFFICIAL_HORIZONS:
            self.assertEqual(
                set(self.service_metrics.loc[self.service_metrics["horizon"].eq(horizon), "model"]),
                set(MODELS_LIST),
            )
        sort = ["model", "horizon", "split"]
        pd.testing.assert_frame_equal(
            self.service_metrics.sort_values(sort).reset_index(drop=True),
            self.backend_metrics.sort_values(sort).reset_index(drop=True),
            check_exact=True,
        )

    def test_naive_is_na_and_other_directional_accuracy_is_official(self):
        display = _format_metrics_for_display(self.service_metrics)
        naive_raw = self.service_metrics.loc[self.service_metrics["model"].eq("Naive")]
        naive_display = display.loc[display["Mô hình"].eq("Naive")]
        self.assertEqual(len(naive_raw), 5)
        self.assertTrue(naive_raw["directional_accuracy"].isna().all())
        self.assertEqual(naive_display["Directional Accuracy (%)"].tolist(), ["N/A"] * 5)
        self.assertTrue(pd.api.types.is_float_dtype(self.service_metrics["directional_accuracy"]))
        np.testing.assert_array_equal(display["MAE (USD)"], self.service_metrics["mae"])
        np.testing.assert_array_equal(display["RMSE (USD)"], self.service_metrics["rmse"])
        for display_row, raw_row in zip(display.itertuples(index=False), self.service_metrics.itertuples(index=False)):
            if raw_row.model != "Naive":
                self.assertEqual(display_row[-1], f"{raw_row.directional_accuracy:.2f}%")
        da_candidates = self.service_metrics.dropna(subset=["directional_accuracy"])
        self.assertEqual(set(da_candidates["model"]), set(MODELS_LIST) - {"Naive"})

    def test_horizon_and_model_filters_match_official_metrics(self):
        for horizon in OFFICIAL_HORIZONS:
            for model in MODELS_LIST:
                with self.subTest(model=model, horizon=horizon):
                    subset = self.predictions.loc[
                        self.predictions["horizon"].eq(horizon) & self.predictions["model"].eq(model)
                    ]
                    self.assertEqual(len(subset), 442)
                    metric = compute_evaluation_metrics(subset, [model])
                    reference = self.backend_metrics.loc[
                        self.backend_metrics["horizon"].eq(horizon) & self.backend_metrics["model"].eq(model)
                    ]
                    pd.testing.assert_frame_equal(
                        metric.reset_index(drop=True), reference.reset_index(drop=True), check_exact=True
                    )

    def test_actual_vs_predicted_chart_uses_official_test_rows(self):
        for horizon in OFFICIAL_HORIZONS:
            for model in MODELS_LIST:
                with self.subTest(model=model, horizon=horizon):
                    subset = self.predictions.loc[
                        self.predictions["horizon"].eq(horizon) & self.predictions["model"].eq(model)
                    ].sort_values("target_date")
                    figure = plot_bento_test_comparison(
                        subset, [model], focus_model=model, zoom_sessions=442
                    )
                    self.assertEqual(len(figure.data), 2)
                    self.assertEqual(list(figure.data[0].x), subset["target_date"].tolist())
                    self.assertEqual(list(figure.data[1].x), subset["target_date"].tolist())
                    np.testing.assert_array_equal(figure.data[0].y, subset["predicted"])
                    np.testing.assert_array_equal(figure.data[1].y, subset["actual"])
        display = _format_metrics_for_display(self.service_metrics.loc[self.service_metrics["horizon"].eq(2)])
        self.assertEqual(len(plot_bento_error_bars(display).data), 2)

    def test_empty_and_missing_data_are_explicit(self):
        empty = self.predictions.iloc[0:0]
        metrics = compute_evaluation_metrics(empty, ["Naive"])
        self.assertTrue(metrics.empty)
        self.assertEqual(metrics.columns.tolist(), list(METRIC_COLUMNS))
        with self.assertRaisesRegex(ValueError, "thiếu cột"):
            compute_evaluation_metrics(self.predictions.drop(columns=["actual"]), MODELS_LIST)


if __name__ == "__main__":
    unittest.main()
