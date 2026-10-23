import unittest

import numpy as np
import pandas as pd

from evaluate_manual_labels import (FEATURES, ai_opening_eligible, chronological_split,
                                    choose_training_threshold, counts,
                                    experimental_score, score_at_cutoff,
                                    trial_candidate_at_10s, trial_geometry_failures,
                                    summarize_trial_history,
                                    trial_signal_eligible, visible_trial_score)


class ManualLabelEvaluationTests(unittest.TestCase):
    def test_ai_opening_scope_has_no_gap_or_angle_cutoff(self):
        row = pd.Series({"gap_pct": .3, "distance_outside_prior_target_pct": .1,
                         "target_angle_deg_5m_equiv": 70.0})
        self.assertTrue(ai_opening_eligible(row))
        row["distance_outside_prior_target_pct"] = -.01
        self.assertFalse(ai_opening_eligible(row))
        row["distance_outside_prior_target_pct"] = .1
        row["opening_gap_direction"] = "DOWN"
        row["target_band"] = "UPPER"
        self.assertFalse(ai_opening_eligible(row))

    def test_trial_score_keeps_range_and_excludes_missing_signal(self):
        row = pd.Series({"gap_pct": 1.5, "distance_outside_target_pct": .7,
                         "model_score_without_filters_10s": 92.0})
        self.assertTrue(trial_signal_eligible(row))
        self.assertEqual(visible_trial_score(row), 92.0)
        inside = row.copy()
        inside["distance_outside_target_pct"] = -.2
        self.assertFalse(trial_signal_eligible(inside))
        self.assertTrue(np.isnan(visible_trial_score(inside)))
        small_gap = row.copy()
        small_gap["gap_pct"] = .1
        self.assertTrue(np.isnan(visible_trial_score(small_gap)))

    def test_previous_score_feature_uses_only_cutoff_inputs(self):
        self.assertIn("existing_score_at_cutoff", FEATURES)
        self.assertNotIn("existing_final_score", FEATURES)
        early = pd.Series({"gap_pct": 1.5, "distance_outside_target_pct": .7,
                           "prior_width_pct": 1.0,
                           "prior_width_change_pp_per_bar": -.01,
                           "target_angle_deg_5m_equiv": 10.0,
                           "existing_final_score": 20})
        later = early.copy()
        later["existing_final_score"] = 95
        self.assertEqual(score_at_cutoff(early), score_at_cutoff(later))

    def test_chronological_split_keeps_sessions_together(self):
        rows = []
        for day in range(1, 9):
            for ticker, positive in (("A", 0), ("B", 1)):
                rows.append({"session_date": f"2026-01-{day:02d}",
                             "ticker": ticker, "positive": positive})
        train, test = chronological_split(pd.DataFrame(rows))
        self.assertTrue(set(train.session_date).isdisjoint(test.session_date))
        self.assertLess(train.session_date.max(), test.session_date.min())


    def test_counts_includes_negative_class_and_threshold_is_training_only(self):
        actual = pd.Series([1, 1, 0, 0])
        result = counts(actual, pd.Series([1, 0, 1, 0]))
        self.assertEqual(result["tp_buenas"], 1)
        self.assertEqual(result["fn_buenas"], 1)
        self.assertEqual(result["tn_malas"], 1)
        self.assertEqual(result["fp_malas"], 1)
        threshold = choose_training_threshold(
            actual, pd.Series([.9, .8, .2, .1]).to_numpy())
        self.assertTrue(.2 < threshold <= .8)

    def test_threshold_prioritizes_rejecting_reviewed_bad_cases(self):
        labels = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0])
        probabilities = np.array([.9, .8, .7, .5, .4, .8, .65, .2, .1])
        threshold = choose_training_threshold(labels, probabilities)
        self.assertGreater(threshold, .65)
        self.assertLessEqual(threshold, .7)

    def test_experimental_score_preserves_seventy_threshold_without_saturation(self):
        scores = experimental_score(np.array([0, .2, .4, .7, 1]), .4)
        self.assertAlmostEqual(scores[0], 0)
        self.assertAlmostEqual(scores[2], 70)
        self.assertTrue(70 < scores[3] < 100)
        self.assertAlmostEqual(scores[4], 100)

    def test_trial_candidate_needs_real_exterior_gap_and_angle(self):
        row = pd.Series({"gap_pct": 1.5, "distance_outside_target_pct": .7,
                         "prior_width_pct": 1.0,
                         "prior_width_change_pp_per_bar": -.01,
                         "target_angle_deg_5m_equiv": 10.0,
                         "experimental_score_10s": 80.0})
        self.assertTrue(trial_candidate_at_10s(row))
        self.assertEqual(trial_geometry_failures(row), [])
        too_steep = row.copy()
        too_steep["target_angle_deg_5m_equiv"] = 35.0
        self.assertEqual(trial_geometry_failures(too_steep),
                         ["Ángulo sobre el límite"])
        self.assertFalse(trial_candidate_at_10s(too_steep))
        inside = row.copy()
        inside["distance_outside_target_pct"] = 0
        self.assertFalse(trial_candidate_at_10s(inside))
        missing = row.copy()
        missing["target_angle_deg_5m_equiv"] = np.nan
        self.assertFalse(trial_candidate_at_10s(missing))

    def test_daily_summary_keeps_missing_cutoff_in_monitored_total(self):
        frame = pd.DataFrame([
            {"session_date": "2026-01-02", "early_data_status": "Disponible",
             "existing_score_at_cutoff": 60, "experimental_score_10s": 80,
             "trial_candidate": True, "score_difference_10s": 20,
             "manual_label": "Buena", "existing_final_score": 75},
            {"session_date": "2026-01-02", "early_data_status": "Sin 10 s",
             "existing_score_at_cutoff": np.nan, "experimental_score_10s": np.nan,
             "trial_candidate": False, "score_difference_10s": np.nan,
             "manual_label": "Pendiente", "existing_final_score": 20},
        ])
        result = summarize_trial_history(frame).iloc[0]
        self.assertEqual(result.monitored, 2)
        self.assertEqual(result.with_10s, 1)
        self.assertEqual(result.trial_candidates, 1)
        self.assertEqual(result.manual_good_candidates, 1)
        self.assertEqual(result.manual_bad_candidates, 0)


if __name__ == "__main__":
    unittest.main()
