"""Reproduce decisiones de apertura con los minutos ya cerrados, sin datos futuros."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR
from opening_scan import _provisional_bands, normalized_band_angle_deg, opening_setups
from scanner_settings import MAX_UPPER_ANGLE_DEG


SNAPSHOTS = DATA_DIR / "snapshots"
MINUTES_PATH = SNAPSHOTS / "early_minutes_2026-10-09.csv"
PRIOR_PATH = SNAPSHOTS / "early_prior_15m_2026-10-09.csv"
OUTPUT_PATH = SNAPSHOTS / "early_window_replay_2026-10-09.csv"


def replay(minutes: pd.DataFrame, prior_bars: pd.DataFrame) -> pd.DataFrame:
    records = []
    for day in sorted(set(minutes.day) & set(prior_bars.day)):
        prior_day = prior_bars.loc[prior_bars.day == day].drop(columns="day")
        minute_day = minutes.loc[minutes.day == day]
        first = minute_day.loc[minute_day.time == "09:30"]
        opening_prices = dict(zip(first.ticker, first.open))
        setups = opening_setups(prior_day, opening_prices=opening_prices)
        for setup in setups.itertuples():
            if setup.candidate_at_open != True:
                continue
            ticker = setup.ticker
            prior = prior_day.loc[prior_day.ticker == ticker]
            observed = minute_day.loc[minute_day.ticker == ticker].set_index("time")
            if observed.empty or "09:30" not in observed.index:
                continue
            sign = 1 if setup.gap_direction == "UP" else -1
            prior_target = (float(setup.prior_bb_upper) if sign == 1 else
                            float(setup.prior_bb_lower))
            first_touch = None
            for offset in range(15):
                minute = f"09:{30 + offset:02d}"
                if minute not in observed.index:
                    continue
                bar = observed.loc[minute]
                hit = (float(bar.low) <= prior_target if sign == 1 else
                       float(bar.high) >= prior_target)
                if hit:
                    first_touch = offset
                    break
            previous_provisional = prior_target
            for elapsed in range(6):
                known = [f"09:{30 + i:02d}" for i in range(elapsed)]
                if not all(minute in observed.index for minute in known):
                    continue
                price = (float(setup.opening_price) if elapsed == 0 else
                         float(observed.loc[known[-1]].close))
                outside = sign * (price / prior_target - 1) > 0
                untouched = first_touch is None or first_touch >= elapsed
                if elapsed:
                    upper, lower = _provisional_bands(prior, price)
                    provisional = upper if sign == 1 else lower
                    angle_from_open = sign * normalized_band_angle_deg(
                        prior_target, provisional, elapsed)
                    angle_latest_minute = sign * normalized_band_angle_deg(
                        previous_provisional, provisional, 1)
                    previous_provisional = provisional
                else:
                    angle_from_open = angle_latest_minute = np.nan
                eligible = outside and untouched
                future_touch = first_touch is not None and first_touch >= elapsed
                records.append({
                    "date": day, "ticker": ticker, "direction": setup.gap_direction,
                    "elapsed_minutes": elapsed, "decision_time": f"09:{30 + elapsed:02d}",
                    "gap_pct": setup.gap_pct,
                    "distance_at_open_pct": setup.distance_to_band_pct,
                    "score_open": setup.score_open,
                    "prior_target_band": prior_target,
                    "price_at_decision": price,
                    "angle_from_open_deg": angle_from_open,
                    "angle_latest_minute_deg": angle_latest_minute,
                    "still_outside_and_untouched": eligible,
                    "passes_angle_25": bool(eligible and elapsed > 0 and
                                            angle_from_open <= MAX_UPPER_ANGLE_DEG),
                    "first_fixed_band_touch": (
                        f"09:{30 + first_touch:02d}" if first_touch is not None else ""),
                    "future_fixed_band_touch_before_0945": bool(eligible and future_touch),
                    "lead_minutes_to_touch": (first_touch - elapsed if eligible and future_touch
                                              else np.nan),
                })
    return pd.DataFrame.from_records(records)


def main() -> None:
    minutes = pd.read_csv(MINUTES_PATH, dtype={"time": str, "day": str})
    prior = pd.read_csv(PRIOR_PATH, dtype={"time": str, "day": str})
    result = replay(minutes, prior)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}: {len(result)} checkpoint rows")
    print("Dates:", result.date.nunique(), "open candidates:",
          result.loc[result.elapsed_minutes == 0, "ticker"].count())
    for elapsed, group in result.groupby("elapsed_minutes"):
        eligible = group.loc[group.still_outside_and_untouched]
        angle = eligible.loc[eligible.passes_angle_25]
        print(f"09:{30 + elapsed:02d}: eligible={len(eligible)}, "
              f"future_touch={eligible.future_fixed_band_touch_before_0945.sum()}, "
              f"angle<=25={len(angle)}, "
              f"angle_future_touch={angle.future_fixed_band_touch_before_0945.sum()}")
    print("CVS today:")
    print(result.loc[(result.date == "2026-10-09") & (result.ticker == "CVS"),
                     ["decision_time", "gap_pct", "score_open", "price_at_decision",
                      "angle_from_open_deg", "angle_latest_minute_deg",
                      "still_outside_and_untouched", "passes_angle_25",
                      "first_fixed_band_touch", "lead_minutes_to_touch"]].to_string(index=False))


if __name__ == "__main__":
    main()
