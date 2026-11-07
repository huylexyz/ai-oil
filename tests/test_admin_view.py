"""Check that the administration page reports official data and artifact status."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
# Import every page before mocked service calls so later AppTest runs retain their real bindings.
from src.views import admin_view, evaluation_view, forecast_view, overview_view


class AdminViewTests(unittest.TestCase):
    @staticmethod
    def _app():
        return AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=90)

    def _run_admin(self):
        app = self._app().run()
        app.sidebar.radio[0].set_value("Quản trị & Kiểm toán MLOps").run()
        self.assertFalse(app.exception)
        return app

    @staticmethod
    def _html(app):
        return "\n".join(str(item.value) for item in app.markdown)

    def test_official_kpis_and_no_realtime_claim(self):
        app = self._run_admin()
        html = self._html(app)
        self.assertIn("8,840 bản ghi Test", html)
        self.assertIn("15 / 15 artifact AI", html)
        self.assertIn("20 tổ hợp", html)
        self.assertIn("442 ngày mục tiêu", html)
        self.assertIn("t+5: 4/4 mẫu", html)
        self.assertNotIn("3,536", html)
        self.assertNotIn("suy luận thời gian thực", html.lower())

    def test_inventory_and_split_counts_match_source_files(self):
        tables = []
        with patch.object(admin_view, "render_genesis_table", side_effect=lambda frame: tables.append(frame.copy())):
            self._run_admin()
        self.assertEqual(len(tables), 3)
        inventory, audit, splits = tables
        self.assertEqual(len(inventory), 20)
        self.assertEqual(len(audit), 20)
        self.assertEqual(inventory.loc[inventory["Mô hình"].eq("Naive"), "Kích thước bộ nhớ"].tolist(), ["Trực tiếp"] * 5)
        self.assertTrue(inventory.loc[inventory["Mô hình"].ne("Naive"), "Trạng thái kiểm định"].eq("ĐẠT").all())
        self.assertTrue(audit["Số phiên Test chuẩn"].eq(442).all())
        self.assertTrue(audit["Trạng thái"].eq("ĐẠT").all())

        for h in range(1, 6):
            feature_path = (
                PROJECT_ROOT / "data" / "processed" / "brent_features.csv" if h in (1, 5)
                else PROJECT_ROOT / "data" / "processed" / f"brent_features_h{h}.csv"
            )
            source = pd.read_csv(feature_path)
            counts = source[f"split_{h}"].value_counts()
            metadata = json.loads((PROJECT_ROOT / "tv5" / "models" / f"rf_model_t{h}_metadata.json").read_text(encoding="utf-8"))
            expected = (
                counts["train"], counts["val"], metadata["trainval_rows_used_for_final_fit"], counts["test"]
            )
            self.assertEqual([row[f"Kỳ hạn t+{h}"] for _, row in splits.iterrows()], [f"{value:,} mẫu" for value in expected])
        self.assertTrue(splits["Trạng thái"].eq("ĐẠT").all())

    def test_missing_artifact_is_not_reported_ready(self):
        candidates = dict(admin_view.MODEL_FILE_CANDIDATES)
        candidates[("Ridge", 3)] = [PROJECT_ROOT / "missing-ridge-h3.joblib"]
        tables = []
        with patch.object(admin_view, "MODEL_FILE_CANDIDATES", candidates):
            with patch.object(admin_view, "render_genesis_table", side_effect=lambda frame: tables.append(frame.copy())):
                app = self._run_admin()
        self.assertIn("14 / 15 artifact AI", self._html(app))
        row = tables[0].loc[tables[0]["Mô hình"].eq("Ridge") & tables[0]["Kỳ hạn"].eq("t+3")].iloc[0]
        self.assertEqual(row["Trạng thái kiểm định"], "Thiếu artifact")
        self.assertNotEqual(row["Trạng thái vận hành"], "Sẵn sàng")

    def test_missing_predictions_and_split_metadata_are_not_success(self):
        with patch.object(admin_view, "get_all_predictions", return_value=(pd.DataFrame(), False, [])):
            with patch.object(admin_view, "FEATURES_DATA_PATH", PROJECT_ROOT / "missing-features.csv"):
                app = self._run_admin()
        html = self._html(app)
        self.assertIn("0 bản ghi Test", html)
        self.assertIn("0/20 tập kết quả", html)
        self.assertIn("Chưa xác minh purge", html)
        self.assertIn("Thiếu nguồn", html)
        self.assertNotIn("8,840 bản ghi Test", html)

    def test_missing_clean_brent_is_not_operational(self):
        with patch.object(admin_view, "get_clean_data", return_value=(pd.DataFrame(), False)):
            app = self._run_admin()
        self.assertEqual(len(app.error), 1)
        self.assertIn("Chưa tìm thấy", str(app.error[0].value))
        self.assertNotIn("8,840 bản ghi Test", self._html(app))
        self.assertNotIn("15 / 15 artifact AI", self._html(app))

    def test_brent_read_exception_does_not_crash_admin_or_sidebar(self):
        with patch("src.services.data_service.get_clean_data", side_effect=OSError("bad file")):
            with patch.object(overview_view, "get_clean_data", side_effect=OSError("bad file")):
                with patch.object(admin_view, "get_clean_data", side_effect=OSError("bad file")):
                    app = self._run_admin()
        self.assertEqual(len(app.error), 1)
        self.assertIn("bad file", str(app.error[0].value))
        self.assertNotIn("8,840 bản ghi Test", self._html(app))
        sidebar = "\n".join(str(item.value) for item in app.sidebar.markdown)
        self.assertIn("Giá khóa sổ (không khả dụng)", sidebar)

    def test_invalid_brent_columns_dates_and_prices_are_reported(self):
        cases = (
            (pd.DataFrame({"observation_date": ["2026-09-30"]}), "thiếu cột"),
            (pd.DataFrame({"observation_date": ["not a date"], "price": [115.91]}), "Không thể hiển thị"),
            (pd.DataFrame({"observation_date": ["2026-09-30"], "price": [float("nan")]}), "bị thiếu"),
            (pd.DataFrame({"observation_date": ["2026-09-30"], "price": [-1.0]}), "hữu hạn và dương"),
        )
        for frame, message in cases:
            with self.subTest(message=message):
                with patch.object(admin_view, "get_clean_data", return_value=(frame, True)):
                    app = self._run_admin()
                self.assertEqual(len(app.error), 1)
                self.assertIn(message, str(app.error[0].value))
                self.assertNotIn("8,840 bản ghi Test", self._html(app))

    def test_navigation_away_from_admin_error(self):
        app = self._app().run()
        self.assertFalse(app.exception)
        with patch.object(admin_view, "get_clean_data", side_effect=OSError("bad file")):
            app.sidebar.radio[0].set_value("Quản trị & Kiểm toán MLOps").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        for page in (
            "Tổng quan & Chuỗi thời gian",
            "Dự báo Kỳ hạn t+1 & t+5",
            "So sánh & Phân tích Sai số",
        ):
            with self.subTest(page=page):
                app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(app.exception)

    def test_four_page_navigation_smoke(self):
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
