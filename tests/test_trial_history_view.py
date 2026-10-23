import unittest

import pandas as pd

from app import balanced_ai_review_sample, trial_review_cases, trial_session_setups


class TrialHistoryViewTests(unittest.TestCase):
    def test_manual_review_uses_twenty_second_model_history(self):
        setups = pd.DataFrame([{"session_date": "2026-10-15", "ticker": "AAA",
                                "score": 30.0, "gap_pct": 1.1}])
        trial = pd.DataFrame([{
            "session_date": "2026-10-15", "ticker": "AAA",
            "model_score_20s": 88.0, "existing_score_at_20s": 41.0,
            "trial_candidate_20s": True, "early_data_status": "Disponible",
            "gap_pct": 1.5, "target_band": "UPPER",
            "distance_outside_target_pct": .8,
            "distance_outside_prior_target_pct": 1.0,
            "target_angle_deg_5m_equiv": 7.0,
            "trial_signal_status_20s": "Señal modelo 20 s",
        }])
        shown = trial_review_cases(setups, trial, model_20s=True)
        self.assertEqual(shown.review_score_trial.iloc[0], 88.0)
        self.assertEqual(shown.review_score_actual_10s.iloc[0], 41.0)
        self.assertTrue(shown.review_trial_candidate.iloc[0])

    def test_twenty_second_model_drives_trial_list_and_chart_cutoff(self):
        originals = pd.DataFrame([
            {"ticker": "AAA", "score": 35.0, "candidate": False,
             "opening_price": 100.0, "gap_pct": 1.0, "cutoff_seconds": 10,
             "cutoff_bb_upper": 101.0, "cutoff_bb_lower": 98.0},
            {"ticker": "BBB", "score": 90.0, "candidate": True,
             "opening_price": 50.0, "gap_pct": 2.0, "cutoff_seconds": 10,
             "cutoff_bb_upper": 51.0, "cutoff_bb_lower": 48.0},
        ])
        model_rows = pd.DataFrame([
            {"ticker": "AAA", "model_score_20s": 84.0,
             "trial_candidate_20s": True, "price": 102.5,
             "gap_pct": 1.7, "upper_provisional": 101.5,
             "lower_provisional": 98.5,
             "distance_outside_target_pct": 0.99,
             "target_angle_deg_5m_equiv": 5.0,
             "early_data_status": "Disponible"},
            {"ticker": "BBB", "model_score_20s": 43.0,
             "trial_candidate_20s": False, "price": 51.0,
             "gap_pct": 2.0, "upper_provisional": 52.0,
             "lower_provisional": 49.0,
             "distance_outside_target_pct": -1.92,
             "target_angle_deg_5m_equiv": 40.0,
             "early_data_status": "Disponible"},
        ])
        shown = trial_session_setups(originals, model_rows,
                                     score_list=True, model_20s=True)
        self.assertEqual(shown.ticker.tolist(), ["AAA", "BBB"])
        self.assertEqual(shown.candidate.tolist(), [True, False])
        self.assertEqual(shown.score.tolist(), [84.0, 43.0])
        self.assertEqual(shown.loc[0, "cutoff_seconds"], 20)
        self.assertEqual(shown.loc[0, "opening_price"], 102.5)
        self.assertEqual(shown.loc[0, "cutoff_bb_upper"], 101.5)
        self.assertEqual(shown.loc[0, "gap_pct"], 1.7)
        limited = trial_session_setups(originals, model_rows,
                                       score_list=True, model_20s=True,
                                       max_signals=0)
        self.assertEqual(int(limited.candidate.sum()), 0)
        self.assertEqual(int(limited.trial_candidate.sum()), 1)
        gap_filtered = trial_session_setups(originals, model_rows,
                                            score_list=True, model_20s=True,
                                            min_gap_pct=2.0)
        self.assertEqual(int(gap_filtered.candidate.sum()), 0)

    def test_manual_review_uses_trial_candidates_and_scores(self):
        setups = pd.DataFrame([
            {"session_date": "2026-10-09", "ticker": "AAA", "score": 45.0,
             "candidate": False, "gap_pct": 2.0},
            {"session_date": "2026-10-09", "ticker": "BBB", "score": 80.0,
             "candidate": True, "gap_pct": 1.0},
        ])
        trial = pd.DataFrame([
            {"session_date": "2026-10-09", "ticker": "AAA",
             "experimental_score_10s": 90.0, "existing_score_at_cutoff": 48.0,
             "trial_candidate": True, "early_data_status": "Disponible",
             "gap_pct": 1.8, "distance_outside_target_pct": 1.2,
             "target_angle_deg_5m_equiv": 8.0},
            {"session_date": "2026-10-09", "ticker": "BBB",
             "experimental_score_10s": 50.0, "existing_score_at_cutoff": 75.0,
             "trial_candidate": False, "early_data_status": "Disponible",
             "gap_pct": .8, "distance_outside_target_pct": .4,
             "target_angle_deg_5m_equiv": 40.0},
        ])
        cases = trial_review_cases(setups, trial)
        self.assertEqual(cases.loc[cases.review_trial_candidate, "ticker"].tolist(), ["AAA"])
        self.assertEqual(cases.loc[cases.review_score_high, "ticker"].tolist(), ["AAA"])
        self.assertEqual(cases.loc[0, "review_score_trial"], 90.0)
        self.assertEqual(cases.loc[0, "review_gap_10s_pct"], 1.8)
        self.assertEqual(cases.loc[0, "score"], 45.0)

    def test_overlays_trial_score_and_candidate_without_changing_original(self):
        originals = pd.DataFrame([
            {"ticker": "AAA", "score": 82.0, "candidate": True, "gap_pct": 2.1},
            {"ticker": "BBB", "score": 45.0, "candidate": False, "gap_pct": 1.0},
            {"ticker": "CCC", "score": 30.0, "candidate": False, "gap_pct": -.2},
        ])
        trial = pd.DataFrame([
            {"ticker": "AAA", "experimental_score_10s": 55.0,
             "model_score_without_filters_10s": 55.0,
             "existing_score_at_cutoff": 61.0, "score_difference_10s": -6.0,
             "gap_pct": 1.8, "distance_outside_target_pct": .7,
             "target_angle_deg_5m_equiv": 12.0, "manual_label": "Mala",
             "early_data_status": "Disponible", "trial_candidate": False},
            {"ticker": "BBB", "experimental_score_10s": 88.0,
             "model_score_without_filters_10s": 88.0,
             "existing_score_at_cutoff": 50.0, "score_difference_10s": 38.0,
             "gap_pct": 1.2, "distance_outside_target_pct": .8,
             "target_angle_deg_5m_equiv": -5.0, "manual_label": "Buena",
             "early_data_status": "Disponible", "trial_candidate": True},
        ])
        shown = trial_session_setups(originals, trial)
        self.assertEqual(shown.ticker.tolist(), ["BBB", "AAA", "CCC"])
        self.assertEqual(shown.score.tolist()[:2], [88.0, 55.0])
        self.assertEqual(shown.candidate.tolist(), [True, False, False])
        self.assertEqual(shown.loc[shown.ticker.eq("AAA"), "gap_pct"].iloc[0], 2.1)
        self.assertEqual(originals.score.tolist(), [82.0, 45.0, 30.0])
        self.assertEqual(originals.candidate.tolist(), [True, False, False])

    def test_score_list_shows_high_scoring_signals_even_if_strict_filters_fail(self):
        originals = pd.DataFrame([{"ticker": "AAA", "score": 40.0,
                                   "candidate": False}])
        trial = pd.DataFrame([{"ticker": "AAA", "experimental_score_10s": 84.8,
                               "model_score_without_filters_10s": 84.8,
                               "existing_score_at_cutoff": 70.0,
                               "score_difference_10s": 14.8,
                               "gap_pct": 3.6, "distance_outside_target_pct": 2.1,
                               "target_angle_deg_5m_equiv": 80.0,
                               "manual_label": "Pendiente",
                               "early_data_status": "Disponible",
                               "trial_signal_status": "No cumple filtros del scanner",
                               "trial_candidate": False}])
        shown = trial_session_setups(originals, trial, score_list=True)
        self.assertTrue(shown.candidate.iloc[0])
        self.assertFalse(shown.trial_candidate.iloc[0])
        self.assertEqual(shown.score.iloc[0], 84.8)

    def test_ai_review_sample_includes_low_middle_and_high_unlabeled_cases(self):
        frame = pd.DataFrame([{
            "session_date": "2026-10-09", "ticker": f"T{i:02d}",
            "gap_direction": "UP", "review_target_band": "UPPER",
            "early_data_status": "Disponible", "review_prior_distance_10s_pct": .5,
            "review_score_trial": float(i),
        } for i in range(1, 31)])
        labeled = {("2026-10-09", "T30"): "Excelente"}
        sample = balanced_ai_review_sample(frame, labeled)
        self.assertEqual(len(sample), 12)
        self.assertNotIn("T30", sample.ticker.tolist())
        self.assertTrue((sample.review_score_trial < 10).any())
        self.assertTrue(sample.review_score_trial.between(10, 20).any())
        self.assertTrue((sample.review_score_trial > 20).any())


if __name__ == "__main__":
    unittest.main()
