"""Evalúa proxies de apertura y 1 minuto frente a score_15m >= 70.

La evaluación con operaciones SIP reales de 10 segundos está separada en
evaluate_historical_10s.py.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATABASE_PATH


FEATURES_OPEN = (
    "abs_gap_pct", "distance_to_band_pct", "prior_bb_width_pct",
    "directional_prior_slope", "prior_bb_width_change",
)
FEATURES_1M = FEATURES_OPEN + (
    "first_minute_reversal_pct", "target_angle_1m_deg",
)
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "snapshots"


def load_examples(database: Path = DATABASE_PATH,
                  today: str | None = None) -> pd.DataFrame:
    """Lee sólo sesiones terminadas; conserva únicamente aperturas exteriores."""
    today = today or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    with closing(sqlite3.connect(database)) as db:
        rows = [(day, json.loads(payload)) for day, payload in db.execute(
            "SELECT session_date, payload_json FROM daily_diagnostics "
            "WHERE session_date < ? ORDER BY session_date, ticker", (today,))]
    records = []
    for day, row in rows:
        score_later = row.get("score_15m")
        distance = row.get("distance_to_band_pct")
        if score_later is None or distance is None or distance <= 0:
            continue
        direction = 1 if row.get("gap_direction") == "UP" else -1
        record = {"date": day, "ticker": row["ticker"],
                  "label_15m_ge_70": int(score_later >= 70),
                  "score_15m": score_later, "score_open": row.get("score_open"),
                  "score_1m": row.get("score_1m"),
                  "abs_gap_pct": abs(row["gap_pct"]),
                  "distance_to_band_pct": distance,
                  "prior_bb_width_pct": row.get("prior_bb_width_pct"),
                  "directional_prior_slope": (direction * row["prior_target_band_slope_pct"]
                                              if row.get("prior_target_band_slope_pct") is not None
                                              else np.nan),
                  "prior_bb_width_change": row.get("prior_bb_width_change"),
                  "first_minute_reversal_pct": row.get("first_minute_reversal_pct"),
                  "target_angle_1m_deg": row.get("target_angle_1m_deg")}
        records.append(record)
    return pd.DataFrame.from_records(records)


def chronological_split(frame: pd.DataFrame, test_days: int,
                        min_train_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = sorted(frame.date.unique())
    if len(days) < min_train_days + test_days:
        raise ValueError(f"Sólo hay {len(days)} sesiones; se necesitan "
                         f"{min_train_days + test_days} para esta evaluación")
    boundary = days[-test_days]
    train = frame.loc[frame.date < boundary].copy()
    test = frame.loc[frame.date >= boundary].copy()
    if train.label_15m_ge_70.nunique() != 2:
        raise ValueError("El entrenamiento necesita ejemplos de ambas clases")
    assert set(train.date).isdisjoint(test.date)
    return train, test


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame,
                  features: tuple[str, ...]) -> np.ndarray:
    """Ajusta imputación y escala sólo sobre el entrenamiento."""
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000, C=1.0))
    model.fit(train.loc[:, features], train.label_15m_ge_70)
    return model.predict_proba(test.loc[:, features])[:, 1]


def classification_counts(labels: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    labels = labels.astype(bool)
    predicted = predicted.astype(bool)
    true_positive = int((labels & predicted).sum())
    false_positive = int((~labels & predicted).sum())
    false_negative = int((labels & ~predicted).sum())
    return {"avisos": int(predicted.sum()), "aciertos": true_positive,
            "falsos_positivos": false_positive, "omitidos": false_negative,
            "precision": round(true_positive / max(true_positive + false_positive, 1), 3),
            "recall": round(true_positive / max(true_positive + false_negative, 1), 3)}


def top_n_counts(frame: pd.DataFrame, column: str, n: int = 3) -> dict[str, float | int]:
    selected = frame.sort_values(["date", column, "ticker"],
                                 ascending=[True, False, True]).groupby("date").head(n)
    return classification_counts(frame.label_15m_ge_70,
                                 frame.index.isin(selected.index))


def evaluate(frame: pd.DataFrame, features: tuple[str, ...],
             test_days: int, min_train_days: int) -> tuple[dict, pd.DataFrame]:
    train, test = chronological_split(frame, test_days, min_train_days)
    test["model_probability"] = fit_and_score(train, test, features)
    labels = test.label_15m_ge_70
    result = {
        "train_dates": [train.date.min(), train.date.max()],
        "test_dates": [test.date.min(), test.date.max()],
        "train_rows": len(train), "test_rows": len(test),
        "train_positives": int(train.label_15m_ge_70.sum()),
        "test_positives": int(labels.sum()),
        "test_prevalence": round(float(labels.mean()), 3),
        "features": list(features),
        "average_precision_model": round(float(average_precision_score(
            labels, test.model_probability)), 3),
        "average_precision_score_open": round(float(average_precision_score(
            labels, test.score_open)), 3),
        "brier_model": round(float(brier_score_loss(labels, test.model_probability)), 3),
        "model_at_0_5": classification_counts(labels, test.model_probability >= 0.5),
        "score_open_at_70": classification_counts(labels, test.score_open >= 70),
        "model_top_3_per_day": top_n_counts(test, "model_probability"),
        "score_open_top_3_per_day": top_n_counts(test, "score_open"),
    }
    if features == FEATURES_1M:
        result["average_precision_score_1m"] = round(float(
            average_precision_score(labels, test.score_1m)), 3)
        result["score_1m_at_70"] = classification_counts(labels, test.score_1m >= 70)
        result["score_1m_top_3_per_day"] = top_n_counts(test, "score_1m")
    return result, test


def main() -> None:
    frame = load_examples()
    if frame.empty:
        raise SystemExit("No hay aperturas históricas con etiqueta final.")
    days = sorted(frame.date.unique())[-50:]
    frame = frame.loc[frame.date.isin(days)].copy()
    ten_second_files = sorted(OUTPUT_DIR.glob("apertura_10s_*.csv"))
    ten_second_rows = sum(len(pd.read_csv(path)) for path in ten_second_files)
    historical_ten_second_files = sorted(OUTPUT_DIR.glob("apertura_historica_10s_*.csv"))
    historical_ten_second_rows = sum(len(pd.read_csv(path))
                                     for path in historical_ten_second_files)
    open_result, open_predictions = evaluate(frame, FEATURES_OPEN, 15, 35)
    minute = frame.dropna(subset=["first_minute_reversal_pct", "target_angle_1m_deg"])
    one_min_result, one_min_predictions = evaluate(minute, FEATURES_1M, 2, 5)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    for name, predictions in (("open_proxy", open_predictions), ("one_minute", one_min_predictions)):
        predictions.to_csv(OUTPUT_DIR / f"early_score_{name}_{stamp}.csv", index=False)
    report = {
        "generated_at_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "label": "score_15m >= 70; semejanza geométrica, no ganancia ni toque",
        "scope": "aperturas exteriores a la banda; sólo símbolos/sesiones guardados",
        "ten_second_snapshot_days": len(ten_second_files),
        "ten_second_quote_rows": ten_second_rows,
        "historical_sip_ten_second_days": len(historical_ten_second_files),
        "historical_sip_ten_second_symbols": historical_ten_second_rows,
        "twenty_thirty_second_historical_examples": 0,
        "ten_second_model_status": (
            "ver evaluación separada: evaluate_historical_10s.py"
            if historical_ten_second_rows else
            "sin evaluar: faltan sesiones subminuto suficientes"),
        "open_proxy_warning": "OPEN histórico de vela; no cotización observada a los 10 segundos",
        "available_days": len(days), "available_rows": len(frame),
        "outside_band_symbols_by_day": frame.groupby("date").ticker.nunique().to_dict(),
        "opening_proxy": open_result, "one_minute": one_min_result,
    }
    path = OUTPUT_DIR / f"early_score_validation_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReporte guardado: {path}")


if __name__ == "__main__":
    main()
