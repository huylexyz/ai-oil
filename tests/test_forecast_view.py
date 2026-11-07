"""The existing forecast page must display direct h1–h5 predictions."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from src.services.data_service import get_clean_data
from src.services.forecast_service import get_all_latest_forecasts
from src.ui.charts import plot_bento_october_forecast
from src.views import forecast_view
from src.views.forecast_view import _build_forecast_table, _validated_forecast_rows


class ForecastViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = ["Naive", "Ridge", "Random Forest", "XGBoost"]
        cls.frame = _validated_forecast_rows(get_all_latest_forecasts(cls.models), cls.models)
        cls.history, has_clean = get_clean_data()
        assert has_clean

    def test_table_contains_all_twenty_official_prices(self):
        table = _build_forecast_table(self.frame, self.models)
        self.assertEqual(len(table), 4)
        self.assertEqual(table["Mô hình"].tolist(), self.models)
        self.assertEqual(table.shape[1], 13)
        for model in self.models:
            row = table.loc[table["Mô hình"].eq(model)].iloc[0]
            for horizon in range(1, 6):
                price = self.frame.loc[
                    self.frame["model"].eq(model) & self.frame["horizon"].eq(horizon),
                    "predicted_price",
                ].iloc[0]
                self.assertEqual(row[f"Dự báo t+{horizon} ($)"], f"${price:.2f}")

    def test_chart_uses_observation_date_and_relative_session_positions(self):
        chart_data = self.frame.pivot(index="horizon", columns="model", values="predicted_price").reset_index()
        history = self.history.tail(25)
        fig = plot_bento_october_forecast(history, chart_data, self.models)
        self.assertEqual(len(fig.data), 5)  # Actual plus four models.
        self.assertEqual(list(fig.data[0].x), list(range(-24, 1)))
        self.assertIn("2026-09-30", list(fig.data[0].customdata))
        for index, model in enumerate(self.models, start=1):
            trace = fig.data[index]
            self.assertEqual(list(trace.x), [0, 1, 2, 3, 4, 5])
            self.assertEqual(list(trace.customdata)[1:], [f"Phiên t+{h}" for h in range(1, 6)])
            np.testing.assert_array_equal(
                list(trace.y)[1:],
                chart_data[model].tolist(),
            )
        self.assertTrue(all(f"t+{h}" in fig.layout.xaxis.ticktext for h in range(1, 6)))
        self.assertNotIn("target_date", chart_data.columns)

    def test_model_selection_limits_chart_and_table_consistently(self):
        models = ["Ridge", "XGBoost"]
        selected = _validated_forecast_rows(self.frame.loc[self.frame["model"].isin(models)].copy(), models)
        table = _build_forecast_table(selected, models)
        chart_data = selected.pivot(index="horizon", columns="model", values="predicted_price").reset_index()
        fig = plot_bento_october_forecast(self.history.tail(15), chart_data, models)
        self.assertEqual(table["Mô hình"].tolist(), models)
        self.assertEqual([trace.name for trace in fig.data[1:]], [f"Dự báo: {m}" for m in models])

    def test_missing_or_inconsistent_rows_are_reported(self):
        frame = self.frame.copy()
        with self.assertRaisesRegex(ValueError, "không trả DataFrame"):
            _validated_forecast_rows(None, self.models)
        missing = frame.loc[~(frame["model"].eq("Ridge") & frame["horizon"].eq(3))]
        with self.assertRaisesRegex(ValueError, "thiếu, trùng"):
            _validated_forecast_rows(missing, self.models)
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "thiếu, trùng"):
            _validated_forecast_rows(duplicate, self.models)
        wrong_date = frame.copy()
        wrong_date.loc[0, "observation_date"] = "2026-09-29"
        with self.assertRaisesRegex(ValueError, "Ngày quan sát"):
            _validated_forecast_rows(wrong_date, self.models)
        invalid_price = frame.copy()
        invalid_price.loc[0, "predicted_price"] = np.nan
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            _validated_forecast_rows(invalid_price, self.models)

    @staticmethod
    def _app():
        return AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=90)

    def test_apptest_normal_forecast_page(self):
        app = self._app().run()
        app.sidebar.radio[0].set_value("Dự báo Kỳ hạn t+1 & t+5").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        text = "\n".join(str(item.value) for item in app.main.markdown)
        self.assertIn("Dự báo t+2 ($)", text)
        self.assertIn("Dự báo t+3 ($)", text)
        self.assertIn("Dự báo t+4 ($)", text)

    def test_apptest_brent_read_exception_stops_before_forecast(self):
        app = self._app().run()
        with patch("src.services.data_service.get_clean_data", side_effect=OSError("bad Brent file")):
            with patch.object(forecast_view, "get_clean_data", side_effect=OSError("bad Brent file")):
                app.sidebar.radio[0].set_value("Dự báo Kỳ hạn t+1 & t+5").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn("bad Brent file", str(app.error[0].value))
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertNotIn("Dự báo t+2 ($)", "\n".join(str(item.value) for item in app.main.markdown))

    def test_apptest_empty_invalid_brent_and_missing_artifact(self):
        cases = (
            (pd.DataFrame(), False),
            (pd.DataFrame({"observation_date": ["2026-09-30"]}), True),
            (pd.DataFrame({"observation_date": ["bad date"], "price": [115.91]}), True),
            (pd.DataFrame({"observation_date": ["2026-09-30"], "price": [float("nan")]}), True),
        )
        for frame, available in cases:
            with self.subTest(available=available, columns=frame.columns.tolist()):
                app = self._app().run()
                with patch.object(forecast_view, "get_clean_data", return_value=(frame, available)):
                    app.sidebar.radio[0].set_value("Dự báo Kỳ hạn t+1 & t+5").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.error), 1)
                self.assertEqual(len(app.get("plotly_chart")), 0)
        app = self._app().run()
        with patch.object(forecast_view, "get_available_joblib_models", side_effect=FileNotFoundError("missing model")):
            app.sidebar.radio[0].set_value("Dự báo Kỳ hạn t+1 & t+5").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn("missing model", str(app.error[0].value))
        self.assertEqual(len(app.get("plotly_chart")), 0)


if __name__ == "__main__":
    unittest.main()
