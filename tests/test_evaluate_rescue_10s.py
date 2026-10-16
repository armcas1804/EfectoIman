import unittest

import pandas as pd

from evaluate_rescue_10s import expanding_date_folds, rescue_examples


class RescueTenSecondTests(unittest.TestCase):
    def test_rescue_group_and_reversal_use_only_early_prices(self):
        frame = pd.DataFrame({
            "date": ["2026-10-09"] * 3,
            "ticker": ["UP", "DOWN", "INSIDE"],
            "distance_to_band_pct_10s": [0.5, 0.6, -0.1],
            "score_10s_geometry": [45, 65, 30],
            "direction_10s": ["UP", "DOWN", "UP"],
            "price_first_10s": [100, 100, 100],
            "price_at_10s": [99, 101, 100],
            "trade_count": [10, 20, 1],
            "last_trade_odd_lot": [False, True, False],
        })
        result = rescue_examples(frame)
        self.assertEqual(result.ticker.tolist(), ["UP", "DOWN"])
        self.assertAlmostEqual(result.iloc[0].early_reversal_pct, 1)
        self.assertAlmostEqual(result.iloc[1].early_reversal_pct, 1)

    def test_expanding_folds_keep_future_dates_out_of_training(self):
        frame = pd.DataFrame({
            "date": [f"{n:02}" for n in range(1, 12) for _ in range(2)],
            "label_15m_ge_70": [0, 1] * 11,
        })
        folds = expanding_date_folds(frame, first_train_days=3, validation_days=2)
        self.assertEqual(len(folds), 4)
        for train, valid in folds:
            self.assertLess(train.date.max(), valid.date.min())
            self.assertTrue(set(train.date).isdisjoint(valid.date))


if __name__ == "__main__":
    unittest.main()
