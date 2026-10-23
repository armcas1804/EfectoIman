"""Compara scores tempranos sobre aperturas realmente evaluables a 10 s.

El estudio no cambia el scanner ni promociona automáticamente un modelo.
Las etiquetas son una muestra elegida para revisión, no un censo del mercado.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_manual_labels import (FEATURES, OUTPUT_DIR, choose_training_threshold,
                                    chronological_split, counts, labeled_examples,
                                    trial_signal_eligible)
from scanner_settings import MAX_UPPER_ANGLE_DEG


def models() -> dict:
    return {
        "Logística regularizada": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=.25, max_iter=2000, class_weight="balanced")),
        "Bosque actual, entrenado sólo con señales": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=150, max_depth=3,
                                   min_samples_leaf=5, class_weight="balanced",
                                   random_state=42)),
        "Árboles aleatorios simples": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=150, max_depth=3,
                                 min_samples_leaf=5, class_weight="balanced",
                                 random_state=42)),
    }


def fit(model, frame: pd.DataFrame) -> None:
    weight = np.where(frame.label.eq("Excelente"), 2.0, 1.0)
    model.fit(frame[list(FEATURES)], frame.positive,
              **{f"{model.steps[-1][0]}__sample_weight": weight})


def chronological_predictions(model, training: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(training.session_date.unique())
    first = max(4, len(dates) // 3)
    step = max(1, len(dates) // 7)
    folds = []
    for stop in range(first, len(dates), step):
        earlier = training.loc[training.session_date.isin(dates[:stop])]
        later = training.loc[training.session_date.isin(dates[stop:stop + step])].copy()
        if earlier.positive.nunique() < 2 or later.empty:
            continue
        fit(model, earlier)
        later["probability"] = model.predict_proba(later[list(FEATURES)])[:, 1]
        folds.append(later)
    return pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()


def study() -> tuple[dict, pd.DataFrame]:
    _, examples = labeled_examples()
    observed = examples.loc[examples.window_seconds.eq(10)].copy()
    observed["signal_at_10s"] = observed.apply(trial_signal_eligible, axis=1)
    eligible = observed.loc[observed.signal_at_10s].copy()
    train, test = chronological_split(eligible)
    report = {
        "generated_at_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "population": "Casos con juicio visual, gap mínimo y precio fuera de Bollinger a 10 s",
        "label_warning": "Los juicios no son una muestra aleatoria de las 72 empresas. "
                         "Estas métricas no determinan pesos ideales para el universo completo.",
        "labels_with_10s": int(len(observed)),
        "labels_with_signal_at_10s": int(len(eligible)),
        "excluded_before_model_by_label": observed.loc[
            ~observed.signal_at_10s, "label"].value_counts().to_dict(),
        "angle_above_limit_among_signals_by_label": {
            label: {"above": int(group.target_angle_deg_5m_equiv.gt(
                        MAX_UPPER_ANGLE_DEG).sum()), "total": len(group)}
            for label, group in eligible.groupby("label")},
        "train_dates": [train.session_date.min(), train.session_date.max()],
        "test_dates": [test.session_date.min(), test.session_date.max()],
        "train_labels": train.label.value_counts().to_dict(),
        "test_labels": test.label.value_counts().to_dict(),
        "features_known_at_10s": list(FEATURES),
        "excellent_training_weight": 2.0,
        "baseline_score_heuristico_10s_ge70": counts(
            test.positive, test.existing_score_at_cutoff.ge(70)),
        "models": {},
    }
    predictions = test[["session_date", "ticker", "label", "positive", "gap_pct",
                        "distance_outside_target_pct", "target_angle_deg_5m_equiv",
                        "existing_score_at_cutoff"]].copy()
    for name, model in models().items():
        calibration = chronological_predictions(model, train)
        if calibration.empty or calibration.positive.nunique() < 2:
            report["models"][name] = {"status": "sin_calibracion_temporal"}
            continue
        threshold = choose_training_threshold(
            calibration.positive, calibration.probability.to_numpy())
        oof = counts(calibration.positive, calibration.probability.ge(threshold))
        fit(model, train)
        probability = model.predict_proba(test[list(FEATURES)])[:, 1]
        predicted = probability >= threshold
        key = {"Logística regularizada": "logistica",
               "Bosque actual, entrenado sólo con señales": "bosque",
               "Árboles aleatorios simples": "arboles"}[name]
        predictions[f"{key}_score_bruto"] = (probability * 100).round(1)
        predictions[f"{key}_seleccionado"] = predicted
        estimator = model.steps[-1][1]
        effects = (estimator.coef_[0] if hasattr(estimator, "coef_") else
                   estimator.feature_importances_)
        report["models"][name] = {
            "threshold_chosen_on_training": round(float(threshold), 3),
            "chronological_training_oof": oof,
            "later_dates": counts(test.positive, predicted),
            "excellent_detected_later": int((test.label.eq("Excelente") & predicted).sum()),
            "excellent_total_later": int(test.label.eq("Excelente").sum()),
            "roc_auc_later": round(float(roc_auc_score(test.positive, probability)), 3),
            "feature_effects_exploratory": {
                feature: round(float(effect), 3)
                for feature, effect in zip(FEATURES, effects)},
            "feature_effect_kind": ("coeficiente_estandarizado" if hasattr(estimator, "coef_")
                                    else "importancia_de_arboles"),
        }
    return report, predictions


def main() -> None:
    report, predictions = study()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    report_path = OUTPUT_DIR / f"ten_second_scoring_study_{stamp}.json"
    prediction_path = OUTPUT_DIR / f"ten_second_scoring_study_{stamp}.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions.to_csv(prediction_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Resultados: {report_path}\nCasos: {prediction_path}")


if __name__ == "__main__":
    main()
