"""Evalúa cotizaciones SIP observadas hasta 09:30:10 contra score_15m.

La etiqueta describe el score posterior del scanner, no una ganancia bursátil.
Las operaciones históricas prueban disponibilidad por hora de mercado, no
latencia de entrega a una instalación en vivo.
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
from sklearn.metrics import average_precision_score, brier_score_loss

from config import DATABASE_PATH, DATA_DIR
from evaluate_early_score import (chronological_split, classification_counts,
                                  fit_and_score, top_n_counts)
from indicators import BOLLINGER_DEVIATIONS, BOLLINGER_WINDOW, bollinger_bands
from opening_scan import _linear_slope, _score, normalized_band_angle_deg


SNAPSHOTS = DATA_DIR / "snapshots"
FEATURES_NO_ANGLE = ("abs_gap_pct_10s", "distance_to_band_pct_10s",
                     "prior_bb_width_pct", "prior_bb_width_change")
FEATURES = FEATURES_NO_ANGLE + ("directional_prior_slope_pct",
                                "target_angle_10s_deg")


def prior_band_dynamics(prior_closes: np.ndarray, current_price: float,
                        prior_upper: float, prior_lower: float,
                        direction: str) -> tuple[float, float]:
    """Pendiente previa y banda 15m provisional usando sólo precio hasta 10 s."""
    if len(prior_closes) < BOLLINGER_WINDOW + 4:
        return np.nan, np.nan
    prior = bollinger_bands(pd.DataFrame({
        "bar_index": range(len(prior_closes)), "close": prior_closes}))
    upper, lower = prior.iloc[-1][["bb_upper", "bb_lower"]]
    if not (np.isclose(upper, prior_upper, rtol=1e-5) and
            np.isclose(lower, prior_lower, rtol=1e-5)):
        return np.nan, np.nan
    sign = 1 if direction == "UP" else -1
    target_values = prior.bb_upper.tail(5) if sign == 1 else prior.bb_lower.tail(5)
    target = prior_upper if sign == 1 else prior_lower
    prior_slope = sign * _linear_slope(target_values, target)
    provisional_closes = np.append(prior_closes[-(BOLLINGER_WINDOW - 1):],
                                   current_price)
    middle = provisional_closes.mean()
    deviation = provisional_closes.std(ddof=0)
    provisional = middle + sign * BOLLINGER_DEVIATIONS * deviation
    angle = sign * normalized_band_angle_deg(target, provisional, 10 / 60)
    return prior_slope, angle


def load_joined_examples(folder: Path = SNAPSHOTS,
                         database: Path = DATABASE_PATH) -> pd.DataFrame:
    paths = sorted(folder.glob("apertura_historica_10s_*.csv"))
    if not paths:
        raise ValueError("No hay archivos apertura_historica_10s_*.csv")
    trades = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if trades.duplicated(["date", "ticker"]).any():
        raise ValueError("Cotizaciones de 10 s duplicadas por fecha y símbolo")
    records = []
    prior_closes = {}
    with closing(sqlite3.connect(database)) as db:
        for day, ticker, payload in db.execute(
                "SELECT session_date, ticker, payload_json FROM daily_diagnostics"):
            row = json.loads(payload)
            records.append({"date": day, "ticker": ticker,
                            "score_15m": row.get("score_15m"),
                            "prior_close": row.get("prior_close"),
                            "prior_bb_upper": row.get("prior_bb_upper"),
                            "prior_bb_lower": row.get("prior_bb_lower"),
                            "prior_bb_width_pct": row.get("prior_bb_width_pct"),
                            "prior_bb_width_change": row.get("prior_bb_width_change"),
                            "stored_open_direction": row.get("gap_direction")})
        for day, ticker, close in db.execute(
                "SELECT session_date, ticker, close FROM daily_candles "
                "WHERE session='PREVIA' ORDER BY session_date, ticker, bar_index"):
            prior_closes.setdefault((day, ticker), []).append(float(close))
    frame = trades.merge(pd.DataFrame.from_records(records), on=["date", "ticker"],
                         how="inner", validate="one_to_one")
    required = ["price_at_10s", "prior_close", "prior_bb_upper", "prior_bb_lower",
                "score_15m"]
    frame = frame.dropna(subset=required).copy()
    for column in required[:-1]:
        frame = frame.loc[frame[column] > 0].copy()
    stamp = pd.to_datetime(frame.last_trade_at_et, utc=True).dt.tz_convert(
        "America/New_York")
    elapsed = (stamp.dt.hour * 3600 + stamp.dt.minute * 60 + stamp.dt.second +
               stamp.dt.microsecond / 1_000_000 - (9 * 3600 + 30 * 60))
    if ((elapsed < 0) | (elapsed > 10)).any():
        raise ValueError("Hay operaciones fuera de los primeros 10 segundos")
    frame["last_trade_elapsed_s"] = elapsed
    frame["gap_pct_10s"] = (frame.price_at_10s / frame.prior_close - 1) * 100
    up = frame.gap_pct_10s >= 0
    frame["direction_10s"] = np.where(up, "UP", "DOWN")
    target = np.where(up, frame.prior_bb_upper, frame.prior_bb_lower)
    frame["distance_to_band_pct_10s"] = np.where(
        up, (frame.price_at_10s / target - 1) * 100,
        (target / frame.price_at_10s - 1) * 100)
    frame["abs_gap_pct_10s"] = frame.gap_pct_10s.abs()
    dynamics = frame.apply(lambda row: prior_band_dynamics(
        np.asarray(prior_closes.get((row.date, row.ticker), []), dtype=float),
        row.price_at_10s, row.prior_bb_upper, row.prior_bb_lower,
        row.direction_10s), axis=1)
    frame[["directional_prior_slope_pct", "target_angle_10s_deg"]] = (
        pd.DataFrame(dynamics.tolist(), index=frame.index))
    frame["label_15m_ge_70"] = (frame.score_15m >= 70).astype(int)
    frame["last_trade_odd_lot"] = frame.last_trade_conditions.apply(
        lambda raw: "I" in json.loads(raw))
    frame["score_10s_geometry"] = frame.apply(
        lambda row: _score({"gap_pct": row.gap_pct_10s,
                            "distance_to_band_pct": row.distance_to_band_pct_10s,
                            "prior_bb_width_pct": row.prior_bb_width_pct,
                            "prior_bb_width_change": row.prior_bb_width_change}, "open"),
        axis=1)
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict:
    labels = frame.label_15m_ge_70
    if frame.empty:
        return {"rows": 0}
    return {
        "rows": len(frame), "positives": int(labels.sum()),
        "prevalence": round(float(labels.mean()), 3),
        "average_precision_model": round(float(average_precision_score(
            labels, frame.model_probability)), 3),
        "average_precision_score_10s": round(float(average_precision_score(
            labels, frame.score_10s_geometry)), 3),
        "brier_model": round(float(brier_score_loss(labels, frame.model_probability)), 3),
        "model_at_0_5": classification_counts(labels, frame.model_probability >= 0.5),
        "score_10s_at_70": classification_counts(labels, frame.score_10s_geometry >= 70),
        "model_top_3_per_day": top_n_counts(frame, "model_probability"),
        "score_10s_top_3_per_day": top_n_counts(frame, "score_10s_geometry"),
    }


def evaluate(frame: pd.DataFrame, test_days: int = 15,
             min_train_days: int = 35) -> tuple[dict, pd.DataFrame]:
    outside = frame.loc[frame.distance_to_band_pct_10s > 0].copy()
    train, test = chronological_split(outside, test_days, min_train_days)
    test["model_probability"] = fit_and_score(train, test, FEATURES)
    test["model_no_angle_probability"] = fit_and_score(
        train, test, FEATURES_NO_ANGLE)
    report = {
        "source_dates": [frame.date.min(), frame.date.max()],
        "source_days": int(frame.date.nunique()),
        "source_rows_with_prior_diagnostics": len(frame),
        "outside_band_rows": len(outside),
        "outside_band_positives": int(outside.label_15m_ge_70.sum()),
        "outside_band_angle_10s_median_deg": round(float(
            outside.target_angle_10s_deg.median()), 2),
        "outside_band_angle_10s_above_25_rows": int((
            outside.target_angle_10s_deg > 25).sum()),
        "train_dates": [train.date.min(), train.date.max()],
        "test_dates": [test.date.min(), test.date.max()],
        "train_rows": len(train),
        "train_positives": int(train.label_15m_ge_70.sum()),
        "features_available_by_0930_10": list(FEATURES),
        "angle_10s_missing_rows": int(frame.target_angle_10s_deg.isna().sum()),
        "all_test": metrics(test),
        "average_precision_model_without_angles": round(float(
            average_precision_score(test.label_15m_ge_70,
                                    test.model_no_angle_probability)), 3),
        "test_without_last_odd_lot": metrics(test.loc[~test.last_trade_odd_lot]),
        "test_same_direction_as_stored_open": metrics(test.loc[
            test.direction_10s == test.stored_open_direction]),
    }
    return report, test


def main() -> None:
    trade_files = sorted(SNAPSHOTS.glob("apertura_historica_10s_*.csv"))
    all_trades = pd.concat((pd.read_csv(path) for path in trade_files),
                           ignore_index=True)
    frame = load_joined_examples()
    report, test = evaluate(frame)
    stamp = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    report.update({
        "generated_at_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "label": "score_15m >= 70; calificación posterior, no rentabilidad",
        "source": "última operación SIP entre 09:30:00 y 09:30:10 ET",
        "historical_trade_files": len(trade_files),
        "historical_trade_rows": len(all_trades),
        "trade_rows_without_prior_diagnostics": len(all_trades) - len(frame),
        "diagnostic_rows_by_day": frame.groupby("date").size().to_dict(),
        "historical_trade_last_odd_lot_rows": int(all_trades.last_trade_conditions.apply(
            lambda raw: "I" in json.loads(raw)).sum()),
        "last_trade_odd_lot_rows": int(frame.last_trade_odd_lot.sum()),
        "outside_band_last_trade_odd_lot_rows": int(frame.loc[
            frame.distance_to_band_pct_10s > 0, "last_trade_odd_lot"].sum()),
        "direction_mismatches": int((frame.direction_10s !=
                                     frame.stored_open_direction).sum()),
        "limitation": ("El precio SIP de la última operación puede ser odd lot y no "
                       "coincidir con OHLC de 15m; timestamps históricos no miden "
                       "latencia de entrega en vivo. Sólo hay diagnósticos previos "
                       "para los símbolos que el scanner guardó cada día."),
        "angle_10s_note": ("El salto instantáneo normalizado a una escala de 5m "
                           "produce ángulos grandes; el límite experimental de "
                           "25 grados para 5m no se aplicó a los 10 segundos."),
    })
    report_path = SNAPSHOTS / f"historical_10s_validation_{stamp}.json"
    predictions_path = SNAPSHOTS / f"historical_10s_predictions_{stamp}.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    test.to_csv(predictions_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReporte: {report_path}\nPredicciones: {predictions_path}")


if __name__ == "__main__":
    main()
