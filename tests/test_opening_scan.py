"""Simetría, cronología y diagnóstico del escáner."""

import math
import unittest

import numpy as np
import pandas as pd

from indicators import bollinger_bands
from opening_scan import (_score, _target_angle_quality, normalized_band_angle_deg, opening_setups,
                          reentry_events, upper_angle_openings)
from real_data import scan_real_day
from scanner_settings import ScannerThresholds


def candles(prior_step: float = 0.0, first: tuple | None = None,
            second: tuple | None = None) -> pd.DataFrame:
    records = []
    for index in range(26):
        price = 100 + index * prior_step
        hour = 9 + (30 + index * 15) // 60
        minute = (30 + index * 15) % 60
        records.append({"ticker": "TEST", "bar_index": index, "session": "PREVIA",
                        "time": f"{hour:02d}:{minute:02d}", "open": price,
                        "high": price, "low": price, "close": price})
    for offset, bar in enumerate((first, second)):
        if bar is not None:
            opening, high, low, close = bar
            records.append({"ticker": "TEST", "bar_index": 26 + offset,
                            "session": "ACTUAL", "time": f"09:{30 + offset * 15}",
                            "open": opening, "high": high, "low": low, "close": close})
    return pd.DataFrame(records)


def minute(opening: float, close: float) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": "TEST", "open": opening,
                          "high": max(opening, close),
                          "low": min(opening, close), "close": close}])


def first_five_early(last_close: float) -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "TEST", "time": f"09:{index:02d}", "open": 102.0,
         "high": max(102.0, last_close), "low": min(102.0, last_close),
         "close": last_close if index == 34 else 101.8}
        for index in range(31, 35)])


class OpeningScanTests(unittest.TestCase):
    def test_prior_width_change_is_percentage_points_per_bar(self) -> None:
        data = candles(first=(102, 103, 101, 102))
        prior = data.session.eq("PREVIA")
        closes = 100 + np.sin(np.arange(26) / 3) * .1 + np.arange(26) * .002
        data.loc[prior, ["open", "high", "low", "close"]] = np.repeat(
            closes[:, None], 4, axis=1)
        bands = bollinger_bands(data.loc[prior])
        widths = ((bands.bb_upper - bands.bb_lower) / bands.bb_middle * 100).tail(5)
        expected = np.polyfit(np.arange(5), widths, 1)[0]
        result = opening_setups(data).iloc[0]
        self.assertNotAlmostEqual(expected, 0)
        self.assertAlmostEqual(result.prior_bb_width_change, expected)

    def test_normalized_angle_uses_actual_elapsed_minutes(self) -> None:
        self.assertAlmostEqual(normalized_band_angle_deg(100, 100.2, 1), 45.0)
        self.assertAlmostEqual(normalized_band_angle_deg(100, 101, 5), 45.0)
        self.assertAlmostEqual(normalized_band_angle_deg(100, 100, 1), 0.0)
        self.assertLess(normalized_band_angle_deg(100, 99.8, 1), 0.0)

    def test_upper_angle_sequence_uses_provisional_15m_band_without_future_data(self) -> None:
        early = pd.DataFrame([
            {"ticker": "TEST", "time": f"09:{start:02d}", "close": 100.1 + (start - 30) * .01}
            for start in range(30, 44)])
        a, sequence_a = upper_angle_openings(
            candles(first=(103, 105, 99, 104)), early, "2026-10-07")
        later = early.copy()
        later.loc[later.time >= "09:35", "close"] += 10
        b, sequence_b = upper_angle_openings(
            candles(first=(103, 105, 99, 99)), later, "2026-10-07")
        row = a.iloc[0]
        closes_0935 = np.array([100.0] * 19 + [float(early.loc[early.time == "09:34", "close"].iloc[0])])
        expected_upper = closes_0935.mean() + 2 * closes_0935.std(ddof=0)
        self.assertAlmostEqual(row.upper_band_0935, expected_upper)
        self.assertAlmostEqual(row.upper_angle_0930_0935,
                               math.degrees(math.atan((expected_upper / 100 - 1) * 100)))
        self.assertEqual(row.upper_band_0935, b.iloc[0].upper_band_0935)
        self.assertEqual(row.upper_angle_0930_0935, b.iloc[0].upper_angle_0930_0935)
        self.assertEqual(sequence_a.loc[sequence_a.time == "09:31", "upper_angle_deg"].iloc[0],
                         sequence_b.loc[sequence_b.time == "09:31", "upper_angle_deg"].iloc[0])
        self.assertEqual(len(sequence_a), 16)  # 09:30 hasta 09:45, inclusive.
        self.assertAlmostEqual(row.upper_band_0945,
                               float(np.array([100.0] * 19 + [104.0]).mean() +
                                     2 * np.array([100.0] * 19 + [104.0]).std(ddof=0)))
        self.assertTrue(row.candidate)
        live, live_series = upper_angle_openings(
            candles(first=(103, 105, 99, 104)), early, "2026-10-07",
            include_final_15m=False)
        self.assertTrue(pd.isna(live.iloc[0].upper_band_0945))
        self.assertNotIn("09:45", live_series.time.tolist())

    def test_upper_angle_filter_rejects_fast_growth_and_missing_minute(self) -> None:
        fast = pd.DataFrame([{"ticker": "TEST", "time": "09:34", "close": 110.0}])
        rejected, _ = upper_angle_openings(candles(first=(103, 104, 102, 103)),
                                           fast, "2026-10-07")
        self.assertGreater(rejected.iloc[0].upper_angle_0930_0935, 25)
        self.assertFalse(rejected.iloc[0].candidate)
        self.assertIn("upper_angle_0930_0935", rejected.iloc[0].failed_conditions)
        missing, _ = upper_angle_openings(candles(first=(103, 104, 102, 103)),
                                          fast.iloc[0:0], "2026-10-07")
        self.assertFalse(missing.iloc[0].candidate)
        self.assertTrue(pd.isna(missing.iloc[0].upper_angle_0930_0935))
        self.assertIn("missing_upper_angle", missing.iloc[0].failed_conditions)

    def test_upper_angle_accepts_negative_angle_and_sorts_by_distance(self) -> None:
        first = candles(first=(105, 106, 104, 105))
        first.loc[first.bar_index == 6, ["open", "high", "close"]] = 103
        second = first.copy()
        second["ticker"] = "OTHER"
        second.loc[second.session == "ACTUAL", "open"] = 104
        minutes = pd.DataFrame([
            {"ticker": ticker, "time": "09:34", "close": 100.0}
            for ticker in ("TEST", "OTHER")])
        result, _ = upper_angle_openings(pd.concat([first, second]), minutes,
                                         "2026-10-07")
        self.assertEqual(result.ticker.tolist(), ["TEST", "OTHER"])
        self.assertTrue(result.candidate.all())
        self.assertTrue((result.upper_angle_0930_0935 < 0).all())
        self.assertTrue((result.opening_price > result.prior_upper_band).all())

    def test_up_and_down_are_symmetric_and_first_minute_is_separate(self) -> None:
        up = opening_setups(candles(), minute(102, 101.8)).iloc[0]
        down = opening_setups(candles(), minute(98, 98.2)).iloc[0]
        self.assertEqual((up.gap_direction, up.target_band, up.expected_reversion_direction),
                         ("UP", "UPPER", "DOWN"))
        self.assertEqual((down.gap_direction, down.target_band, down.expected_reversion_direction),
                         ("DOWN", "LOWER", "UP"))
        self.assertTrue(up.candidate_at_open)
        self.assertTrue(down.candidate_at_open)
        self.assertEqual((up.confirmed, down.confirmed), (1, 1))
        self.assertGreater(up.distance_to_band_pct, 0)
        self.assertGreater(down.distance_to_band_pct, 0)
        self.assertEqual(up.confirmation_time, "09:31")

    def test_diagnostic_retains_failures_and_filtered_mode(self) -> None:
        data = candles(prior_step=.05)
        diagnostic = opening_setups(data, minute(103, 102.8))
        self.assertEqual(len(diagnostic), 1)
        self.assertFalse(diagnostic.iloc[0].candidate)
        self.assertGreater(diagnostic.iloc[0].score, 0)
        self.assertIn("band_slope", diagnostic.iloc[0].failed_conditions)
        self.assertTrue(opening_setups(data, minute(103, 102.8), mode="FILTERED").empty)

    def test_thresholds_can_be_disabled_individually(self) -> None:
        limits = ScannerThresholds(maximum_target_band_slope_pct=None,
                                   maximum_prior_bb_width_pct=None,
                                   maximum_prior_bb_width_growth=None)
        result = opening_setups(candles(prior_step=.05), minute(103, 102.8),
                                thresholds=limits)
        self.assertTrue(result.iloc[0].candidate_at_open)

    def test_current_15m_close_cannot_change_open_or_minute_metrics(self) -> None:
        a = opening_setups(candles(first=(102, 103, 98, 99)), minute(102, 102.1)).iloc[0]
        b = opening_setups(candles(first=(102, 104, 101, 103)), minute(102, 102.1)).iloc[0]
        for key in ("score_open", "score_1m", "candidate_at_open", "confirmed",
                    "prior_bb_width_pct", "distance_to_band_pct"):
            self.assertEqual(a[key], b[key])
        self.assertNotEqual(a.post_open_bb_width_pct, b.post_open_bb_width_pct)

    def test_post_open_fields_are_absent_before_first_15m_close(self) -> None:
        pre = opening_setups(candles(), minute(102, 101.8)).iloc[0]
        post = opening_setups(candles(first=(102, 102.2, 101, 101.5)),
                              minute(102, 101.8)).iloc[0]
        self.assertTrue(pd.isna(pre.post_open_bb_width_pct))
        self.assertTrue(pd.isna(pre.post_open_target_band_slope_pct))
        self.assertTrue(pd.notna(post.post_open_bb_width_pct))
        self.assertEqual(pre.score_open, post.score_open)

    def test_first_minute_reversal_is_not_hard_candidate_filter(self) -> None:
        result = opening_setups(candles(), minute(102, 102.1)).iloc[0]
        self.assertEqual(result.confirmed, 0)
        self.assertTrue(result.candidate)
        self.assertIn("first_minute_confirmation", result.failed_conditions)

    def test_missing_minute_is_pending_not_rejected(self) -> None:
        result = opening_setups(candles(), opening_prices={"TEST": 102}).iloc[0]
        self.assertEqual(result.confirmed, -1)
        self.assertIn("first_minute_confirmation", result.pending_conditions)
        self.assertTrue(result.candidate_at_open)

    def test_ten_second_score_uses_provisional_band_and_marks_missing_angle(self) -> None:
        prior = candles()
        baseline = opening_setups(prior, opening_prices={"TEST": 102}).iloc[0]
        cutoff = opening_setups(prior, opening_prices={"TEST": 102},
                                cutoff_seconds=10).iloc[0]
        self.assertGreater(float(cutoff.cutoff_bb_upper), float(cutoff.prior_bb_upper))
        self.assertLess(float(cutoff.distance_to_band_pct),
                        float(baseline.distance_to_band_pct))
        self.assertEqual(cutoff.score_distance_component < baseline.score_distance_component,
                         True)
        self.assertTrue(pd.isna(cutoff.target_angle_10s_deg))
        self.assertIn("target_angle_10s", cutoff.pending_conditions)
        self.assertNotIn("band_slope", cutoff.failed_conditions)

    def test_ten_second_angle_uses_two_post_open_prices(self) -> None:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        opened = datetime(2026, 10, 13, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        flat = opening_setups(candles(), opening_prices={"TEST": 102},
                              cutoff_seconds=10,
                              opening_events={"TEST": [(opened, 102),
                                                        (opened + timedelta(seconds=8), 102)]}).iloc[0]
        rising = opening_setups(candles(), opening_prices={"TEST": 102},
                                cutoff_seconds=10,
                                opening_events={"TEST": [(opened, 101),
                                                          (opened + timedelta(seconds=8), 102)]}).iloc[0]
        self.assertEqual(flat.target_angle_10s_deg, 0)
        self.assertEqual(flat.angle_10s_event_span_seconds, 8)
        self.assertGreater(rising.target_angle_10s_deg, 25)
        self.assertTrue(flat.candidate_at_open)
        self.assertFalse(rising.candidate_at_open)
        self.assertIn("target_angle_10s", rising.failed_conditions)

    def test_reentry_uses_previously_known_band_and_both_directions(self) -> None:
        up = reentry_events(candles(first=(102, 102.2, 99.9, 101.5)),
                            {"TEST": 25}, {"TEST": "UPPER"})
        down = reentry_events(candles(first=(98, 100.1, 97.8, 98.5)),
                              {"TEST": 25}, {"TEST": "LOWER"})
        self.assertEqual(up.iloc[0].target_band, "UPPER")
        self.assertEqual(down.iloc[0].target_band, "LOWER")
        self.assertEqual(up.iloc[0].time, "09:45")
        self.assertAlmostEqual(float(up.iloc[0].price), 100.0)

    def test_confirmed_signal_does_not_use_opening_bar_range(self) -> None:
        data = candles(first=(102, 102.2, 99, 101), second=(101, 101, 99, 100))
        setup = opening_setups(data, minute(102, 101.8)).iloc[0]
        self.assertEqual(setup.touch_time, "10:00")
        self.assertEqual(setup.minutes_until_touch, 29)
        self.assertEqual(setup.touch_status, "FIRST_OBSERVED_EARLY_UNKNOWN")

    def test_rapid_touch_after_0931_uses_only_closed_minute(self) -> None:
        early = pd.DataFrame([{"ticker": "TEST", "time": "09:31", "open": 101.8,
                               "high": 101.9, "low": 99.9, "close": 100.2}])
        result = opening_setups(candles(first=(102, 102.2, 99.8, 100.1)),
                                minute(102, 101.8), early_minutes=early).iloc[0]
        self.assertEqual(result.touch_time, "09:32")
        self.assertEqual(result.minutes_until_touch, 1)
        self.assertEqual(result.touch_status, "OBSERVED_1M")

    def test_post_open_limit_does_not_change_opening_candidate(self) -> None:
        limits = ScannerThresholds(maximum_post_open_bb_expansion_pct=0.0)
        result = opening_setups(candles(first=(102, 105, 101, 104)),
                                minute(102, 101.8), thresholds=limits).iloc[0]
        self.assertTrue(result.candidate_at_open)
        self.assertFalse(result.candidate)
        self.assertIn("post_open_expansion", result.failed_conditions)

    def test_inside_band_has_no_reentry_event(self) -> None:
        data = candles(first=(100.0, 100.2, 99.8, 100.0))
        for index in range(26):
            price = 99.9 if index % 2 == 0 else 100.1
            data.loc[data.bar_index == index, ["open", "high", "low", "close"]] = price
        result = opening_setups(data).iloc[0]
        self.assertLess(result.distance_to_band_pct, 0)
        self.assertFalse(result.touched_target_band)
        self.assertEqual(result.touch_status, "NOT_APPLICABLE_INSIDE_BAND")

    def test_band_catching_open_is_labeled_separately(self) -> None:
        data = candles(first=(102, 105, 101, 105), second=(105, 105, 101, 102))
        result = opening_setups(data, minute(102, 101.8)).iloc[0]
        self.assertTrue(result.touched_target_band)
        self.assertTrue(result.band_at_or_beyond_open_at_touch)
        self.assertLess(result.reversion_to_touch_band_pct, 0)

    def test_five_min_provisional_band_uses_only_first_five_closed_minutes(self) -> None:
        early = first_five_early(101.5)
        a = opening_setups(candles(first=(102, 105, 99, 104)),
                           minute(102, 101.8), early_minutes=early).iloc[0]
        b = opening_setups(candles(first=(102, 103, 100, 100)),
                           minute(102, 101.8), early_minutes=early).iloc[0]
        closes = np.array([100.0] * 19 + [101.5])
        expected_band = closes.mean() + 2 * closes.std(ddof=0)
        expected_slope = ((expected_band / 100) - 1) * 100 / 5
        self.assertAlmostEqual(a.five_min_band_open, 100.0)
        self.assertAlmostEqual(a.five_min_band_end, expected_band)
        self.assertAlmostEqual(a.five_min_band_slope_pct_per_min, expected_slope)
        self.assertEqual(a.five_min_band_slope_pct_per_min, b.five_min_band_slope_pct_per_min)
        self.assertEqual(a.score_open, b.score_open)
        self.assertEqual(a.score_1m, b.score_1m)
        self.assertFalse(a.candidate)
        self.assertIn("target_angle", a.failed_conditions)

    def test_angle_outranks_larger_gap_and_excludes_expanding_band(self) -> None:
        suitable = opening_setups(candles(), minute(102, 101.8),
                                  early_minutes=first_five_early(100.7)).iloc[0]
        expanding = opening_setups(candles(), minute(104, 103.8),
                                   early_minutes=first_five_early(102.0)).iloc[0]
        self.assertLess(suitable.gap_pct, expanding.gap_pct)
        self.assertLess(suitable.upper_angle_0930_0935, 25)
        self.assertGreater(expanding.upper_angle_0930_0935, 25)
        self.assertGreater(suitable.score, expanding.score)
        self.assertTrue(suitable.candidate)
        self.assertFalse(expanding.candidate)
        self.assertIn("target_angle", expanding.failed_conditions)
        self.assertEqual(suitable.score, suitable.score_5m)

    def test_angle_limit_filters_even_when_old_slope_limit_passes(self) -> None:
        result = opening_setups(candles(), minute(102, 101.8),
                                early_minutes=first_five_early(101.0)).iloc[0]
        self.assertGreater(result.upper_angle_0930_0935, 25)
        self.assertLess(result.five_min_band_slope_pct_per_min, .15)
        self.assertFalse(result.candidate)
        self.assertEqual(result.failed_conditions, "target_angle")

    def test_five_min_slope_above_limit_rejects_without_changing_open_score(self) -> None:
        baseline = opening_setups(candles(), minute(102, 101.8)).iloc[0]
        result = opening_setups(candles(), minute(102, 101.8),
                                early_minutes=first_five_early(102.0)).iloc[0]
        self.assertGreater(result.five_min_band_slope_pct_per_min, .15)
        self.assertTrue(result.candidate_at_open)
        self.assertFalse(result.candidate)
        self.assertIn("five_min_band_slope", result.failed_conditions)
        self.assertEqual(result.score_open, baseline.score_open)

    def test_missing_five_minute_bar_keeps_existing_filter_pending(self) -> None:
        partial = first_five_early(101.5).iloc[:3]
        baseline = opening_setups(candles(), minute(102, 101.8)).iloc[0]
        result = opening_setups(candles(), minute(102, 101.8),
                                early_minutes=partial).iloc[0]
        self.assertTrue(pd.isna(result.five_min_band_slope_pct_per_min))
        self.assertIn("five_min_band_slope", result.pending_conditions)
        self.assertEqual(result.candidate, baseline.candidate)

    def test_moderate_five_min_slope_can_replace_stricter_prior_slope_for_up_gap(self) -> None:
        limits = ScannerThresholds(maximum_prior_bb_width_pct=None,
                                   maximum_prior_bb_width_growth=None)
        baseline = opening_setups(candles(prior_step=.05), minute(103, 102.8),
                                  thresholds=limits).iloc[0]
        result = opening_setups(candles(prior_step=.05), minute(103, 102.8),
                                early_minutes=first_five_early(101.0),
                                thresholds=limits).iloc[0]
        self.assertFalse(baseline.candidate_at_open)
        self.assertIn("band_slope", baseline.failed_conditions)
        self.assertLess(result.five_min_band_slope_pct_per_min, 0)
        self.assertTrue(result.candidate)
        self.assertNotIn("band_slope", result.failed_conditions)

    def test_down_gap_uses_lower_band_angle_with_opposite_sign(self) -> None:
        baseline = opening_setups(candles(), minute(98, 98.2)).iloc[0]
        result = opening_setups(candles(), minute(98, 98.2),
                                early_minutes=first_five_early(99.0)).iloc[0]
        self.assertTrue(pd.isna(result.five_min_band_slope_pct_per_min))
        self.assertTrue(baseline.candidate)
        self.assertGreater(result.target_angle_5m_deg, 25)
        self.assertFalse(result.candidate)
        self.assertIn("target_angle", result.failed_conditions)
        self.assertNotIn("five_min_band_slope", result.pending_conditions)

    def test_target_angle_score_curve(self) -> None:
        for angle, expected in ((-5, 1), (0, 1), (5, .88), (20, .52),
                                (25, .4), (40, 4 / 15), (55, 2 / 15),
                                (70, 0), (80, 0)):
            self.assertAlmostEqual(_target_angle_quality(angle), expected)

    def test_open_score_only_penalizes_large_provisional_band_expansion(self) -> None:
        metrics = {"gap_pct": 2.0, "distance_to_band_pct": 1.0,
                   "prior_bb_width_pct": 1.0, "prior_bb_width_change": 0.0,
                   "target_angle_cutoff_deg": 0.0}
        original = _score(metrics, "open")
        small = {**metrics, "cutoff_bb_upper": 101.0, "cutoff_bb_lower": 99.0}
        large = {**metrics, "cutoff_bb_upper": 102.0, "cutoff_bb_lower": 98.0}
        breakdown = {}
        self.assertEqual(_score(small, "open"), original)
        self.assertAlmostEqual(_score(large, "open", breakdown), original - 20.0)
        self.assertEqual(breakdown["score_cutoff_expansion_component"], -20.0)
        self.assertEqual(_score(metrics, "open"), original)

    def test_flat_separated_band_outranks_expanding_band_despite_first_minute(self) -> None:
        flat = dict(gap_pct=1.75, distance_to_band_pct=1.31, prior_bb_width_pct=1.32,
                    prior_bb_width_change=-2.8, target_angle_1m_deg=74,
                    target_angle_5m_deg=.6, target_angle_0935_0940=0,
                    target_angle_0940_0945=2.1, first_minute_reversal_pct=-.41,
                    post_open_bb_expansion_pct=3.1)
        expanding = dict(gap_pct=1.10, distance_to_band_pct=.79, prior_bb_width_pct=.81,
                         prior_bb_width_change=-7.8, target_angle_1m_deg=48,
                         target_angle_5m_deg=3.7, target_angle_0935_0940=14.5,
                         target_angle_0940_0945=14.1, first_minute_reversal_pct=.23,
                         post_open_bb_expansion_pct=119)
        self.assertGreater(_score(flat, "5m"), _score(expanding, "5m"))
        self.assertGreater(_score(flat, "15m"), _score(expanding, "15m"))

    def test_five_minute_historical_bands_and_score_breakdown(self) -> None:
        five = pd.DataFrame([
            {"ticker": "TEST", "time": "09:30", "close": 100.7},
            {"ticker": "TEST", "time": "09:35", "close": 100.5},
        ])
        a = opening_setups(candles(first=(102, 103, 101, 102)), five_minutes=five).iloc[0]
        b = opening_setups(candles(first=(102, 110, 99, 108)), five_minutes=five).iloc[0]
        self.assertTrue(pd.isna(a.target_angle_1m_deg))
        self.assertAlmostEqual(a.target_angle_5m_deg, a.target_angle_0930_0935)
        self.assertAlmostEqual(a.target_angle_0935_0940,
                               normalized_band_angle_deg(a.five_min_band_end,
                                   np.array([100.0] * 19 + [100.5]).mean() +
                                   2 * np.array([100.0] * 19 + [100.5]).std(ddof=0), 5))
        self.assertTrue(pd.notna(a.target_angle_0940_0945))
        self.assertEqual(a.score_5m, b.score_5m)  # el cierre de 15m aún no existe a las 09:35
        self.assertGreater(a.score_15m, b.score_15m)
        self.assertNotEqual(a.post_open_bb_expansion_pct, b.post_open_bb_expansion_pct)
        components = [a.score_gap_component, a.score_distance_component,
                      a.score_compression_component, a.score_prior_width_growth_component,
                      a.score_target_angle_component, a.score_first_minute_component,
                      a.score_post_open_expansion_component]
        self.assertAlmostEqual(sum(components), a.score, places=1)

    def test_historical_scan_prefers_five_minute_bars_to_recent_one_minute_bars(self) -> None:
        raw = candles(first=(102, 103, 101, 102))
        table = raw[["time", "open", "high", "low", "close"]].copy()
        table["day"] = np.where(raw.session == "PREVIA", "2026-09-29", "2026-09-30")
        five = pd.DataFrame([{"day": "2026-09-30", "time": "09:30",
                              "open": 102, "high": 103, "low": 100, "close": 100.7}])
        ones = pd.DataFrame([{"day": "2026-09-30", "time": f"09:{start:02d}",
                              "open": 102, "high": 103, "low": 100,
                              "close": 102 if start == 34 else 101.8}
                             for start in range(30, 35)])
        result, _ = scan_real_day({"TEST": table}, "2026-09-30",
                                  {"TEST": ones}, {"TEST": five})
        expected = opening_setups(raw, five_minutes=pd.DataFrame([
            {"ticker": "TEST", "time": "09:30", "close": 100.7}])).iloc[0]
        self.assertAlmostEqual(result.iloc[0].target_angle_5m_deg,
                               expected.target_angle_5m_deg)
        self.assertTrue(pd.notna(result.iloc[0].target_angle_1m_deg))

    def test_invalid_session_order_is_rejected(self) -> None:
        data = candles()
        data.loc[data.bar_index == 10, "session"] = "ACTUAL"
        with self.assertRaises(ValueError):
            opening_setups(data, minute(102, 101.8))


if __name__ == "__main__":
    unittest.main()
