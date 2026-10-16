"""Estudio supervisado: rescatar scores <70 a los 10 s que superan 70 a 15m.

Este script es sólo evaluación histórica. No modifica los pesos ni las alertas
de la operativa diaria.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_early_score import (chronological_split, classification_counts,
                                  fit_and_score, top_n_counts)
from evaluate_historical_10s import FEATURES, SNAPSHOTS, load_joined_examples


TRADE_FEATURES = FEATURES + (
    "early_reversal_pct", "log_trade_count", "last_trade_elapsed_s",
    "last_trade_odd_lot", "gap_up_10s")


def rescue_examples(frame: pd.DataFrame) -> pd.DataFrame:
    """La selección del grupo y todas las variables se conocen a los 10 s."""
    frame = frame.loc[(frame.distance_to_band_pct_10s > 0) &
                      (frame.score_10s_geometry < 70)].copy()
    direction = np.where(frame.direction_10s == "UP", 1, -1)
    frame["early_reversal_pct"] = -direction * (
        frame.price_at_10s / frame.price_first_10s - 1) * 100
    frame["log_trade_count"] = np.log1p(frame.trade_count)
    frame["last_trade_odd_lot"] = frame.last_trade_odd_lot.astype(int)
    frame["gap_up_10s"] = (frame.direction_10s == "UP").astype(int)
    return frame


def candidate_models() -> dict:
    def logistic(balanced: bool = False):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=1000, C=0.3,
                                                class_weight=("balanced" if balanced
                                                              else None)))
    return {
        "geometry_logistic": (logistic(), FEATURES),
        "trades_logistic": (logistic(), TRADE_FEATURES),
        "trades_balanced_logistic": (logistic(True), TRADE_FEATURES),
        "trades_shallow_boosting": (
            make_pipeline(SimpleImputer(strategy="median"),
                          HistGradientBoostingClassifier(
                              max_iter=80, learning_rate=0.04, max_leaf_nodes=3,
                              min_samples_leaf=20, l2_regularization=5,
                              random_state=17)), TRADE_FEATURES),
    }


def expanding_date_folds(frame: pd.DataFrame,
                         first_train_days: int = 15,
                         validation_days: int = 4) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    days = sorted(frame.date.unique())
    folds = []
    for start in range(first_train_days, len(days), validation_days):
        validation = days[start:start + validation_days]
        if len(validation) != validation_days:
            break
        train = frame.loc[frame.date.isin(days[:start])]
        valid = frame.loc[frame.date.isin(validation)]
        if train.label_15m_ge_70.nunique() < 2 or valid.label_15m_ge_70.nunique() < 2:
            continue
        folds.append((train, valid))
    return folds


def cross_validate(frame: pd.DataFrame, model, features: tuple[str, ...]) -> dict:
    predictions = []
    for train, valid in expanding_date_folds(frame):
        estimator = clone(model)
        estimator.fit(train.loc[:, features], train.label_15m_ge_70)
        scored = valid[["date", "ticker", "label_15m_ge_70"]].copy()
        scored["probability"] = estimator.predict_proba(valid.loc[:, features])[:, 1]
        predictions.append(scored)
    if not predictions:
        raise ValueError("No hay suficientes fechas con ambas clases para validar")
    result = pd.concat(predictions, ignore_index=True)
    labels = result.label_15m_ge_70
    return {
        "validation_days": int(result.date.nunique()),
        "validation_rows": len(result), "validation_positives": int(labels.sum()),
        "average_precision": round(float(average_precision_score(
            labels, result.probability)), 4),
        "top_3_per_day": top_n_counts(result, "probability"),
    }


def evaluate(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    all_outside = frame.loc[frame.distance_to_band_pct_10s > 0].copy()
    reference_train, _ = chronological_split(all_outside, 15, 35)
    low = rescue_examples(frame)
    train, test = chronological_split(low, test_days=15, min_train_days=35)
    candidates = candidate_models()
    validation = {name: cross_validate(train, model, features)
                  for name, (model, features) in candidates.items()}
    # Se escoge usando exclusivamente las fechas de entrenamiento.
    choice = max(candidates, key=lambda name: (
        validation[name]["average_precision"],
        validation[name]["top_3_per_day"]["aciertos"]))
    model, features = candidates[choice]
    model.fit(train.loc[:, features], train.label_15m_ge_70)
    test["rescue_probability"] = model.predict_proba(test.loc[:, features])[:, 1]
    test["reference_probability"] = fit_and_score(reference_train, test, FEATURES)
    selected = test.sort_values(["date", "rescue_probability", "ticker"],
                                ascending=[True, False, True]).groupby("date").head(3)
    test["suggested_top_3"] = test.index.isin(selected.index)
    labels = test.label_15m_ge_70
    report = {
        "target": "score_15m >= 70 dado score_10s < 70 y apertura fuera de BB",
        "training_dates": [train.date.min(), train.date.max()],
        "test_dates": [test.date.min(), test.date.max()],
        "training_rows": len(train),
        "training_positives": int(train.label_15m_ge_70.sum()),
        "validation_by_model": validation,
        "selected_model": choice,
        "selected_features": list(features),
        "test_rows": len(test), "test_positives": int(labels.sum()),
        "test_prevalence": round(float(labels.mean()), 4),
        "test_average_precision": round(float(average_precision_score(
            labels, test.rescue_probability)), 4),
        "test_brier": round(float(brier_score_loss(
            labels, test.rescue_probability)), 4),
        "test_top_3_per_day": classification_counts(labels, test.suggested_top_3),
        "reference_model_top_3_per_day": top_n_counts(test, "reference_probability"),
        "test_probability_at_0_5": classification_counts(
            labels, test.rescue_probability >= 0.5),
        "caveat": ("El conjunto final de 15 fechas ya fue inspeccionado en el estudio "
                   "anterior; el modelo se elige sólo con validación de fechas "
                   "anteriores. Hay pocas etiquetas positivas. No representa "
                   "rentabilidad ni latencia real de recepción."),
    }
    return report, test


def main() -> None:
    report, predictions = evaluate(load_joined_examples())
    stamp = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    report_path = SNAPSHOTS / f"rescue_10s_validation_{stamp}.json"
    predictions_path = SNAPSHOTS / f"rescue_10s_predictions_{stamp}.csv"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReporte: {report_path}\nPredicciones: {predictions_path}")


if __name__ == "__main__":
    main()
