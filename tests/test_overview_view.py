"""The overview must show official Brent data or an explicit error."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from src.services.data_service import get_clean_data
from src.views import overview_view


class OverviewViewTests(unittest.TestCase):
    @staticmethod
    def _app():
        return AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=90)

    @staticmethod
    def _main_text(app):
        return "\n".join(str(item.value) for item in app.main.markdown)

    def _run_with_clean_data(self, replacement):
        with ExitStack() as stack:
            stack.enter_context(patch("src.services.data_service.get_clean_data", **replacement))
            stack.enter_context(patch.object(overview_view, "get_clean_data", **replacement))
            return self._app().run()

    def test_normal_data_keeps_chart_tables_and_official_last_price(self):
        clean, available = get_clean_data()
        self.assertTrue(available)
        latest = clean.sort_values("observation_date").iloc[-1]
        app = self._app().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        text = self._main_text(app)
        self.assertIn(f"${latest['price']:.2f}", text)
        self.assertIn(pd.to_datetime(latest["observation_date"]).strftime("%Y-%m-%d"), text)
        self.assertIn(f"{len(clean):,} phiên", text)
        self.assertIn("Bảng tra cứu lịch sử giá", text)

    def test_file_read_exception_is_reported_without_chart_or_fake_price(self):
        app = self._run_with_clean_data({"side_effect": FileNotFoundError("brent_clean.csv")})
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn("brent_clean.csv", str(app.error[0].value))
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertNotIn("$115.91", self._main_text(app))
        sidebar = "\n".join(str(item.value) for item in app.sidebar.markdown)
        self.assertIn("Giá khóa sổ (không khả dụng)", sidebar)
        self.assertNotIn("$115.91", sidebar)

    def test_empty_missing_columns_and_invalid_values(self):
        cases = (
            (pd.DataFrame(), False, "Chưa tìm thấy"),
            (pd.DataFrame({"observation_date": ["2026-09-30"], "other": [1.0]}), True, "thiếu cột"),
            (pd.DataFrame({"observation_date": ["not a date"], "price": [115.91]}), True, "Không thể hiển thị"),
            (pd.DataFrame({"observation_date": ["2026-09-30"], "price": [float("nan")]}), True, "bị thiếu"),
            (pd.DataFrame({"observation_date": ["2026-09-30"], "price": [115.91]}), True, "ít nhất hai phiên"),
        )
        for frame, available, message in cases:
            with self.subTest(message=message):
                app = self._run_with_clean_data({"return_value": (frame, available)})
                self.assertFalse(app.exception)
                self.assertEqual(len(app.error), 1)
                self.assertIn(message, str(app.error[0].value))
                self.assertEqual(len(app.get("plotly_chart")), 0)
                self.assertNotIn("$115.91", self._main_text(app))

    def test_missing_features_and_removed_log_do_not_show_fallback_counts(self):
        with patch.object(overview_view, "get_features_data", return_value=(pd.DataFrame(), False)):
            with patch.object(overview_view, "get_removed_rows", return_value=(pd.DataFrame(), False)):
                app = self._app().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        text = self._main_text(app)
        self.assertIn("Thiếu đặc trưng", text)
        self.assertIn("N/A", text)
        self.assertNotIn("đã lọc 82", text)
        self.assertNotIn("442 phiên", text)

    def test_navigation_after_overview_error(self):
        with ExitStack() as stack:
            stack.enter_context(patch("src.services.data_service.get_clean_data", side_effect=OSError("bad file")))
            stack.enter_context(patch.object(overview_view, "get_clean_data", side_effect=OSError("bad file")))
            app = self._app().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            app.sidebar.radio[0].set_value("So sánh & Phân tích Sai số").run()
            self.assertFalse(app.exception)
            sidebar = "\n".join(str(item.value) for item in app.sidebar.markdown)
            self.assertIn("Giá khóa sổ (không khả dụng)", sidebar)

    def test_four_pages_normal_smoke(self):
        app = self._app().run()
        self.assertFalse(app.exception)
        for page in (
            "Dự báo Kỳ hạn t+1 & t+5",
            "So sánh & Phân tích Sai số",
            "Quản trị & Kiểm toán MLOps",
            "Tổng quan & Chuỗi thời gian",
        ):
            with self.subTest(page=page):
                app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
