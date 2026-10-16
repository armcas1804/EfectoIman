"""Contrasta los scores existentes con reingresos observados en el histórico.

Los escenarios de tiempo y movimiento son diagnósticos, no nuevas reglas del
scanner ni etiquetas definitivas para entrenar un modelo.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from config import DATABASE_PATH
from evaluate_historical_10s import SNAPSHOTS, load_joined_examples


def load_audit() -> pd.DataFrame:
    early = load_joined_examples()
    records = []
    with closing(sqlite3.connect(DATABASE_PATH)) as db:
        for day, ticker, payload in db.execute(
                "SELECT session_date, ticker, payload_json FROM daily_diagnostics"):
            row = json.loads(payload)
            records.append({
                "date": day, "ticker": ticker,
                "first_minute_reversal_pct": row.get("first_minute_reversal_pct"),
                "target_angle_5m_deg": row.get("target_angle_5m_deg"),
                "post_open_bb_expansion_pct": row.get("post_open_bb_expansion_pct"),
                "failed_conditions": row.get("failed_conditions"),
                "touched_target_band": row.get("touched_target_band"),
                "touch_time": row.get("touch_time"),
                "touch_status": row.get("touch_status"),
                "minutes_until_touch": row.get("minutes_until_touch"),
                "reversion_to_touch_band_pct": row.get("reversion_to_touch_band_pct"),
            })
    frame = early.merge(pd.DataFrame.from_records(records), on=["date", "ticker"],
                        validate="one_to_one")
    return frame.loc[frame.distance_to_band_pct_10s > 0].copy()


def scenario_counts(frame: pd.DataFrame, minutes: int,
                    minimum_reversion_pct: float) -> dict:
    """Cuenta reingresos; los de la primera vela pueden tener hora imprecisa."""
    observed = (frame.touched_target_band.eq(True) &
                frame.minutes_until_touch.le(minutes) &
                frame.reversion_to_touch_band_pct.ge(minimum_reversion_pct))
    high = frame.score_15m >= 70
    after_cutoff_certain = observed & frame.minutes_until_touch.gt(15)
    return {
        "max_minutes": minutes,
        "minimum_reversion_pct": minimum_reversion_pct,
        "observed_cases": int(observed.sum()),
        "observed_with_score_15m_ge_70": int((observed & high).sum()),
        "observed_with_score_15m_lt_70": int((observed & ~high).sum()),
        "high_score_without_observed_outcome": int((~observed & high).sum()),
        "first_bar_time_ambiguous": int((observed &
                                         frame.minutes_until_touch.le(15)).sum()),
        "after_10s_certain_cases": int(after_cutoff_certain.sum()),
        "after_10s_certain_with_score_15m_lt_70": int((
            after_cutoff_certain & ~high).sum()),
    }


def main() -> None:
    frame = load_audit()
    scenarios = [scenario_counts(frame, minutes, movement)
                 for minutes in (15, 30, 45)
                 for movement in (0.0, 0.5, 1.0)]
    report = {
        "generated_at_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "scope": "aperturas exteriores a la banda a los 10s con diagnóstico posterior",
        "rows": len(frame), "dates": int(frame.date.nunique()),
        "score_15m_ge_70": int((frame.score_15m >= 70).sum()),
        "scenarios": scenarios,
        "limitation": ("El toque de la primera vela de 15m puede haber ocurrido "
                       "antes o después del corte de 10s; su minuto exacto no "
                       "está registrado para todos los días. El porcentaje de "
                       "reversión usa el precio de apertura de la vela histórica, "
                       "no un precio de ejecución."),
    }
    stamp = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    csv_path = SNAPSHOTS / f"score_outcome_audit_{stamp}.csv"
    json_path = SNAPSHOTS / f"score_outcome_audit_{stamp}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nTabla: {csv_path}\nResumen: {json_path}")


if __name__ == "__main__":
    main()
