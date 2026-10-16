"""Registro local de cuándo un diagnóstico estuvo disponible para la aplicación."""

import csv
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import DATA_DIR


NEW_YORK = ZoneInfo("America/New_York")
AUDIT_PATH = DATA_DIR / "signal_availability.csv"
_LOCK = threading.Lock()
_COLUMNS = ("recorded_at_et", "scan_started_at_et", "source", "date", "ticker",
            "signal_stage", "candidate_at_open", "candidate", "score", "score_open",
            "gap_pct", "distance_to_band_pct", "target_angle_1m_deg",
            "target_angle_5m_deg")


def record_availability(day: str, source: str, setups: pd.DataFrame,
                        started_at: datetime) -> None:
    """Anota filas de la lista preliminar o confirmada al concluir el cálculo."""
    if setups.empty:
        return
    interesting = setups.loc[
        setups.get("candidate_at_open", pd.Series(False, index=setups.index)).fillna(False) |
        setups.get("candidate", pd.Series(False, index=setups.index)).fillna(False)]
    if interesting.empty:
        return
    recorded = datetime.now(NEW_YORK).isoformat(timespec="seconds")
    started = started_at.astimezone(NEW_YORK).isoformat(timespec="seconds")
    rows = []
    for setup in interesting.to_dict("records"):
        values = {key: setup.get(key, "") for key in _COLUMNS}
        values.update(recorded_at_et=recorded, scan_started_at_et=started,
                      source=source, date=day)
        rows.append(values)
    with _LOCK:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        exists = AUDIT_PATH.exists() and AUDIT_PATH.stat().st_size > 0
        with AUDIT_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)


def first_availability(day: str) -> dict[str, tuple[str, str]]:
    """Devuelve la primera hora registrada por símbolo para la fecha."""
    if not AUDIT_PATH.exists():
        return {}
    with _LOCK:
        try:
            with AUDIT_PATH.open(newline="", encoding="utf-8") as file:
                records = csv.DictReader(file)
                first = {}
                for row in records:
                    if row["date"] == day and row["ticker"] not in first:
                        first[row["ticker"]] = (row["recorded_at_et"], row["source"])
                return first
        except (OSError, KeyError):
            return {}
