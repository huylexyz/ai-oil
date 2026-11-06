"""Acceptance checks for the TV3 horizon 2/3/4 handoff CSVs."""

import unittest

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.features import FEATURE_COLUMNS, build_features
from src.horizon_data import NEW_HORIZONS, build_horizon_frame


class HorizonDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(PROCESSED_DATA_DIR / "brent_clean.csv", parse_dates=["observation_date"])
        cls.raw = cls.raw.sort_values("observation_date").reset_index(drop=True)
        cls.official = pd.read_csv(PROCESSED_DATA_DIR / "brent_features.csv", parse_dates=["observation_date"])
        cls.frames = {
            h: pd.read_csv(
                PROCESSED_DATA_DIR / f"brent_features_h{h}.csv",
                parse_dates=["observation_date", f"target_date_{h}"],
            )
            for h in NEW_HORIZONS
        }

    def test_approved_feature_schema_and_values(self):
        expected_cols = ["observation_date", *FEATURE_COLUMNS]
        self.assertEqual(len(FEATURE_COLUMNS), 29)
        for h, frame in self.frames.items():
            with self.subTest(horizon=h):
                self.assertEqual(frame.columns.tolist(), [
                    *expected_cols, f"target_date_{h}", f"target_{h}",
                    f"target_ret_{h}", f"split_{h}",
                ])
                joined = frame.merge(
                    self.official[expected_cols], on="observation_date",
                    suffixes=("_new", "_official"), validate="one_to_one",
                )
                self.assertEqual(len(joined), len(frame))
                for feature in FEATURE_COLUMNS:
                    np.testing.assert_allclose(
                        joined[f"{feature}_new"], joined[f"{feature}_official"],
                        rtol=0, atol=1e-12,
                    )
                model_inputs = [c for c in frame if c != "observation_date" and not c.startswith(("target_", "split_"))]
                self.assertEqual(model_inputs, list(FEATURE_COLUMNS))

    def test_independent_session_targets_including_boundaries(self):
        prices = self.raw["DCOILBRENTEU"].to_numpy(dtype=float)
        dates = self.raw["observation_date"].tolist()
        origin_index = {date: i for i, date in enumerate(dates)}
        for h, frame in self.frames.items():
            with self.subTest(horizon=h):
                samples = [0, 150, len(frame) // 2, len(frame) - 1]
                samples += [int(frame.index[frame[f"split_{h}"].eq(split)][0]) for split in ("val", "test")]
                for row in frame.iloc[samples].itertuples(index=False):
                    t = origin_index[row.observation_date]
                    self.assertEqual(getattr(row, f"target_date_{h}"), dates[t + h])
                    self.assertEqual(getattr(row, f"target_{h}"), prices[t + h])
                    self.assertAlmostEqual(
                        getattr(row, f"target_ret_{h}"),
                        np.log(prices[t + h]) - np.log(prices[t]), places=12,
                    )

    def test_split_counts_dates_and_boundary_purge(self):
        for h, frame in self.frames.items():
            with self.subTest(horizon=h):
                self.assertEqual(len(frame), 2933 - h)
                self.assertEqual(frame[f"split_{h}"].value_counts().to_dict(), {
                    "train": 1986 - h, "val": 505, "test": 442,
                })
                train = frame[frame[f"split_{h}"].eq("train")]
                val = frame[frame[f"split_{h}"].eq("val")]
                test = frame[frame[f"split_{h}"].eq("test")]
                target_date = f"target_date_{h}"
                self.assertLessEqual(train[target_date].max(), pd.Timestamp("2022-12-31"))
                self.assertGreater(val[target_date].min(), pd.Timestamp("2022-12-31"))
                self.assertLessEqual(val[target_date].max(), pd.Timestamp("2024-12-31"))
                self.assertGreater(test[target_date].min(), pd.Timestamp("2024-12-31"))
                self.assertEqual((train[target_date] > val.observation_date.min()).sum(), h - 1)
                self.assertEqual((val[target_date] > test.observation_date.min()).sum(), h - 1)
                self.assertTrue((train.loc[train[target_date] <= val.observation_date.min(), target_date] <= val.observation_date.min()).all())
                self.assertTrue((val.loc[val[target_date] <= test.observation_date.min(), target_date] <= test.observation_date.min()).all())

    def test_test_dates_match_official_horizons_and_no_bad_rows(self):
        reference = self.official[self.official["split_1"].eq("test")]["target_date_1"].tolist()
        self.assertEqual(reference, self.official[self.official["split_5"].eq("test")]["target_date_5"].tolist())
        for h, frame in self.frames.items():
            with self.subTest(horizon=h):
                self.assertFalse(frame.isna().any().any())
                self.assertFalse(frame.duplicated(["observation_date", f"target_date_{h}"]).any())
                self.assertTrue((frame["observation_date"] < frame[f"target_date_{h}"]).all())
                self.assertEqual(frame.loc[frame[f"split_{h}"].eq("test"), f"target_date_{h}"].dt.strftime("%Y-%m-%d").tolist(), reference)
                self.assertTrue(np.isfinite(frame[[*FEATURE_COLUMNS, f"target_{h}", f"target_ret_{h}"]].to_numpy(dtype=float)).all())

    def test_generator_and_default_tv3_behavior(self):
        official_columns = self.official.columns.tolist()
        baseline = build_features(self.raw)
        self.assertEqual(baseline.columns.tolist(), official_columns)
        self.assertEqual(len(baseline), len(self.official))
        np.testing.assert_allclose(
            baseline[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            self.official[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            rtol=0, atol=1e-12,
        )
        for old_horizon in (1, 5):
            date_col = f"target_date_{old_horizon}"
            self.assertEqual(
                baseline[date_col].dt.strftime("%Y-%m-%d").fillna("").tolist(),
                self.official[date_col].fillna("").tolist(),
            )
            self.assertEqual(baseline[f"split_{old_horizon}"].tolist(), self.official[f"split_{old_horizon}"].fillna("").tolist())
            for target_col in (f"target_{old_horizon}", f"target_ret_{old_horizon}"):
                np.testing.assert_allclose(
                    baseline[target_col], self.official[target_col],
                    rtol=0, atol=1e-12, equal_nan=True,
                )
        for h, exported in self.frames.items():
            with self.subTest(horizon=h):
                built = build_horizon_frame(self.raw, h)
                self.assertEqual(built.columns.tolist(), exported.columns.tolist())
                self.assertEqual(len(built), len(exported))
                np.testing.assert_allclose(built[f"target_{h}"], exported[f"target_{h}"], rtol=0, atol=1e-12)
                for row in exported.iloc[[0, len(exported) // 2, -1]].itertuples(index=False):
                    truncated = self.raw[self.raw.observation_date.le(row.observation_date)]
                    past_features = build_features(truncated).iloc[-1]
                    np.testing.assert_allclose(
                        np.asarray([getattr(row, c) for c in FEATURE_COLUMNS], dtype=float),
                        past_features[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
                        rtol=0, atol=1e-12,
                    )
        with self.assertRaises(ValueError):
            build_horizon_frame(self.raw, 5)


if __name__ == "__main__":
    unittest.main()
