"""Sidebar must show official Brent observations and honest forecast status."""

import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from src.predict import predict_latest
from src.services.data_service import get_clean_data


class SidebarTests(unittest.TestCase):
    @staticmethod
    def _html(app):
        return "\n".join(str(item.value) for item in app.sidebar.markdown)

    @staticmethod
    def _app():
        return AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=90)

    def test_official_observation_and_predictions(self):
        clean, available = get_clean_data()
        self.assertTrue(available)
        latest = clean.sort_values("observation_date").iloc[-1]
        expected = {h: predict_latest("Random Forest", h) for h in (1, 5)}
        app = self._app().run()
        self.assertFalse(app.exception)
        html = self._html(app)
        self.assertIn(pd.to_datetime(latest["observation_date"]).strftime("%d/%m/%Y"), html)
        self.assertIn(f"${latest['price']:.2f}", html)
        for horizon, result in expected.items():
            self.assertIn(f"Dự báo t+{horizon} (Random Forest)", html)
            self.assertIn(f"${result['predicted_price']:.2f}", html)
        self.assertNotIn("real-time", html.lower())

    def test_api_failure_does_not_substitute_observed_price(self):
        app = self._app()
        with patch("src.services.forecast_service.get_latest_official_forecast", side_effect=RuntimeError("API lỗi")):
            app.run()
        self.assertFalse(app.exception)
        html = self._html(app)
        self.assertEqual(html.count("Lỗi dự báo chính thức"), 2)
        self.assertIn("$115.91", html)
        self.assertEqual(html.count('class="sb-h-price">N/A'), 2)

    def test_one_failed_horizon_does_not_hide_the_other(self):
        def one_fails(model, horizon):
            if horizon == 1:
                raise RuntimeError("h1 unavailable")
            return predict_latest(model, horizon)

        app = self._app()
        with patch("src.services.forecast_service.get_latest_official_forecast", side_effect=one_fails):
            app.run()
        self.assertFalse(app.exception)
        html = self._html(app)
        self.assertEqual(html.count('class="sb-h-price">N/A'), 1)
        self.assertIn(f"${predict_latest('Random Forest', 5)['predicted_price']:.2f}", html)

    def test_missing_and_invalid_brent_data_have_no_hardcoded_values(self):
        cases = [
            (pd.DataFrame(), False),
            (pd.DataFrame({"observation_date": ["2026-09-30"], "price": [float("nan")]}), True),
        ]
        for data in cases:
            with self.subTest(data=data[1]):
                app = self._app()
                with patch("src.services.data_service.get_clean_data", return_value=data):
                    app.run()
                self.assertFalse(app.exception)
                html = self._html(app)
                self.assertNotIn("$115.91", html)
                self.assertNotIn("30/09/2026", html)
                self.assertEqual(html.count('class="sb-h-price">N/A'), 2)
                self.assertIn("Giá khóa sổ (không khả dụng)", html)
                reason = "Dữ liệu Brent không hợp lệ" if data[1] else "Thiếu dữ liệu Brent"
                self.assertEqual(html.count(reason), 2)

    def test_missing_artifact_is_visible_without_fake_forecast(self):
        app = self._app()
        with patch("src.services.forecast_service.get_available_joblib_models", side_effect=FileNotFoundError("artifact")):
            with patch("src.services.forecast_service.get_latest_official_forecast", side_effect=FileNotFoundError("model")):
                app.run()
        self.assertFalse(app.exception)
        html = self._html(app)
        self.assertEqual(html.count("Thiếu model/dữ liệu"), 2)
        self.assertEqual(html.count('class="sb-h-price">N/A'), 2)

    def test_data_read_exception_does_not_crash_navigation(self):
        app = self._app().run()
        with patch("src.services.data_service.get_clean_data", side_effect=OSError("bad file")):
            app.sidebar.radio[0].set_value("So sánh & Phân tích Sai số").run()
        self.assertFalse(app.exception)
        html = self._html(app)
        self.assertEqual(html.count("Dữ liệu Brent không hợp lệ"), 2)
        self.assertEqual(html.count('class="sb-h-price">N/A'), 2)

    def test_navigation_preserves_sidebar_values(self):
        app = self._app().run()
        initial = self._html(app)
        self.assertFalse(app.exception)
        for view in (
            "Dự báo Kỳ hạn t+1 & t+5",
            "So sánh & Phân tích Sai số",
            "Quản trị & Kiểm toán MLOps",
        ):
            with self.subTest(view=view):
                app.sidebar.radio[0].set_value(view).run()
                self.assertFalse(app.exception)
                self.assertEqual(self._html(app), initial)


if __name__ == "__main__":
    unittest.main()
