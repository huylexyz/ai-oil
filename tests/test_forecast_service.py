"""Forecast service must expose only approved, direct horizon predictions."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.core.config import HORIZONS, MODEL_FILE_CANDIDATES, MODELS_LIST, OFFICIAL_HORIZONS
from src.core.validators import validate_prediction_df
from src.predict import MODEL_FILES, predict_latest
from src.services import forecast_service


class ForecastServiceTests(unittest.TestCase):
    def test_configuration_and_prediction_csv_validation(self):
        self.assertEqual(HORIZONS, [1, 5])  # Existing pages remain unchanged in TV7.2.
        self.assertEqual(OFFICIAL_HORIZONS, [1, 2, 3, 4, 5])
        self.assertEqual(len(MODEL_FILE_CANDIDATES), 15)
        sample = pd.DataFrame({
            "observation_date": ["2026-09-30"],
            "target_date": ["2026-10-01"],
            "horizon": [3],
            "actual": [115.0],
            "predicted": [116.0],
            "model": ["Ridge"],
        })
        self.assertTrue(validate_prediction_df(sample, "h3.csv")[0])
        sample["horizon"] = 6
        self.assertFalse(validate_prediction_df(sample, "h6.csv")[0])

    def test_twenty_direct_predictions_match_backend(self):
        frame = forecast_service.get_all_latest_forecasts()
        self.assertEqual(frame.shape, (20, 7))
        self.assertEqual(frame.columns.tolist(), [
            "model", "horizon", "observation_date", "last_price",
            "predicted_price", "absolute_change", "percent_change",
        ])
        self.assertEqual(frame.groupby("model")["horizon"].apply(set).to_dict(), {
            model: set(OFFICIAL_HORIZONS) for model in MODELS_LIST
        })
        self.assertEqual(set(frame["observation_date"]), {"2026-09-30"})
        self.assertEqual(set(frame["last_price"]), {115.91})
        for row in frame.itertuples(index=False):
            with self.subTest(model=row.model, horizon=row.horizon):
                official = predict_latest(row.model, row.horizon)
                np.testing.assert_allclose(row.predicted_price, official["predicted_price"], rtol=0, atol=1e-8)
                np.testing.assert_allclose(row.absolute_change, official["absolute_change"], rtol=0, atol=1e-8)
                np.testing.assert_allclose(row.percent_change, official["percent_change"], rtol=0, atol=1e-8)
        naive = frame.loc[frame["model"].eq("Naive")]
        self.assertTrue(naive["predicted_price"].eq(naive["last_price"]).all())

    def test_session_adapter_uses_direct_prices_without_target_dates(self):
        with patch.object(forecast_service, "get_all_latest_forecasts") as get_all:
            get_all.return_value = pd.DataFrame([
                {
                    "model": model, "horizon": horizon,
                    "observation_date": "2026-09-30", "predicted_price": model_idx * 100 + horizon / 7,
                }
                for model_idx, model in enumerate(MODELS_LIST)
                for horizon in OFFICIAL_HORIZONS
            ])
            frame = forecast_service.generate_october_week_forecast(method="direct")
        self.assertEqual(frame["horizon"].tolist(), OFFICIAL_HORIZONS)
        self.assertNotIn("target_date", frame.columns)
        self.assertEqual(set(frame["observation_date"]), {"2026-09-30"})
        for model_idx, model in enumerate(MODELS_LIST):
            np.testing.assert_array_equal(
                frame[model].to_numpy(),
                [model_idx * 100 + horizon / 7 for horizon in OFFICIAL_HORIZONS],
            )
        with self.assertRaises(ValueError):
            forecast_service.generate_october_week_forecast(method="stepwise")

    def test_backend_error_has_no_inference_fallback(self):
        with patch.object(forecast_service, "backend_predict_latest", side_effect=FileNotFoundError("missing model")):
            with self.assertRaisesRegex(FileNotFoundError, "missing model"):
                forecast_service.get_latest_official_forecast("Ridge", 2)

    def test_failed_horizon_aborts_instead_of_returning_partial_results(self):
        def predict(model, horizon):
            if (model, horizon) == ("Ridge", 3):
                raise FileNotFoundError("ridge h3 missing")
            return {
                "model": model, "horizon": horizon, "observation_date": "2026-09-30",
                "last_price": 115.91, "predicted_price": 115.91,
                "absolute_change": 0.0, "percent_change": 0.0,
            }

        with patch.object(forecast_service, "backend_predict_latest", side_effect=predict):
            with self.assertRaisesRegex(RuntimeError, r"Ridge t\+3.*ridge h3 missing"):
                forecast_service.get_all_latest_forecasts()

    def test_invalid_selections_are_rejected(self):
        for models in ([], ["Unknown"], ["Ridge", "Ridge"], "Ridge"):
            with self.subTest(models=models), self.assertRaises(ValueError):
                forecast_service.get_all_latest_forecasts(models)
        for model, horizon in (("Unknown", 1), ("Ridge", 6), ("Ridge", True), ("Ridge", "2")):
            with self.subTest(model=model, horizon=horizon), self.assertRaises(ValueError):
                forecast_service.get_latest_official_forecast(model, horizon)

    def test_missing_artifact_is_reported(self):
        with patch.dict(MODEL_FILES, {("Ridge", 3): Path("missing-ridge-h3.joblib")}):
            with self.assertRaisesRegex(FileNotFoundError, "missing-ridge-h3.joblib"):
                forecast_service.get_available_joblib_models()


if __name__ == "__main__":
    unittest.main()
