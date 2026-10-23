"""Recalcula únicamente los scores guardados, sin descargar ni cambiar velas."""

import json
import sqlite3
from datetime import datetime

import numpy as np

from config import DATABASE_PATH, UNIVERSE_ID
from opening_scan import _score


SCORE_FIELDS = (
    "score_open", "score_1m", "score_5m", "score_15m", "score",
    "target_angle_score", "score_gap_component", "score_distance_component",
    "score_compression_component", "score_prior_width_growth_component",
    "score_target_angle_component", "score_first_minute_component",
    "score_post_open_expansion_component",
    "score_cutoff_expansion_component",
)


def main() -> None:
    with sqlite3.connect(DATABASE_PATH) as db:
        rows = db.execute(
            "SELECT d.session_date, d.source, d.ticker, d.payload_json "
            "FROM daily_diagnostics d JOIN daily_sessions s "
            "ON s.session_date=d.session_date AND s.source=d.source "
            "WHERE s.universe_id=?", (UNIVERSE_ID,)).fetchall()
        if not rows:
            print("No hay diagnósticos de la versión actual para recalcular.")
            return
        backup_path = DATABASE_PATH.with_name(
            f"efecto_iman_pre_score_{datetime.now():%Y%m%d_%H%M%S}.db")
        with sqlite3.connect(backup_path) as backup:
            db.backup(backup)
        updates = []
        for day, source, ticker, payload in rows:
            record = json.loads(payload)
            metrics = {key: np.nan if value is None else value
                       for key, value in record.items()}
            metrics["score_open"] = _score(metrics, "open", metrics)
            metrics["score_1m"] = (_score(metrics, "1m", metrics)
                                   if np.isfinite(metrics["first_minute_open"]) else np.nan)
            metrics["score_5m"] = (_score(metrics, "5m", metrics)
                                   if np.isfinite(metrics["target_angle_5m_deg"]) else np.nan)
            metrics["score_15m"] = (_score(metrics, "15m", metrics)
                                    if np.isfinite(metrics["post_open_bb_width_pct"]) else np.nan)
            metrics["score"] = next(value for value in
                                    (metrics["score_15m"], metrics["score_5m"],
                                     metrics["score_1m"], metrics["score_open"])
                                    if np.isfinite(value))
            for field in SCORE_FIELDS:
                value = metrics.get(field, np.nan)
                record[field] = float(value) if np.isfinite(value) else None
            updates.append((record["score"], json.dumps(record, ensure_ascii=False,
                                                         allow_nan=False), day, source, ticker))
        with db:
            db.executemany(
                "UPDATE daily_diagnostics SET score=?, payload_json=? "
                "WHERE session_date=? AND source=? AND ticker=?", updates)
        print(f"Scores actualizados: {len(updates)}. Respaldo: {backup_path}")


if __name__ == "__main__":
    main()
