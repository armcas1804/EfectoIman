"""El estudio subminuto no debe usar observaciones posteriores al corte."""

import unittest

import pandas as pd

from study_cutoff_models import feature_table


class CutoffFeaturesTest(unittest.TestCase):
    def test_twenty_second_features_ignore_thirty_second_observation(self):
        common = {
            "session_date": "2026-10-01", "ticker": "XYZ", "target_band": "UPPER",
            "prior_width_pct": 2.0, "upper_provisional": 102.0,
            "lower_provisional": 98.0, "gap_abs_pct": 1.5,
            "distance_outside_target_pct": 0.5,
            "target_angle_deg_5m_equiv": 3.0, "event_count": 10,
        }
        observations = pd.DataFrame([
            {**common, "window_seconds": 10, "price": 103.0},
            {**common, "window_seconds": 20, "price": 102.5},
            {**common, "window_seconds": 30, "price": 999.0},
        ])
        first = feature_table(observations, 20)
        observations.loc[observations.window_seconds.eq(30), "price"] = 1.0
        second = feature_table(observations, 20)

        self.assertEqual(first["price_move_10_to_cutoff_pct"].iloc[0], second[
            "price_move_10_to_cutoff_pct"].iloc[0])
        self.assertEqual(first["provisional_width_ratio"].iloc[0], 2.0)
