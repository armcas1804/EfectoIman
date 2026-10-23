"""Estudio supervisado por corte, sin límites geométricos inventados.

Usa sólo operaciones anteriores a cada corte y etiquetas visuales posteriores.
El conjunto de prueba son fechas posteriores que nunca eligen modelo ni umbral.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline

from evaluate_manual_labels import OUTPUT_DIR, labeled_examples


WINDOWS = (10, 20, 30, 40, 50, 60)
# No usar el score anterior: codifica los límites manuales que queremos estudiar.
FEATURES = (
    "gap_pct", "gap_abs_pct", "distance_outside_target_pct",
    "distance_outside_prior_target_pct", "prior_width_pct",
    "prior_width_change_pp_per_bar", "prior_target_slope_pct_per_bar",
    "target_band_reaction_pct", "target_angle_deg_5m_equiv",
    "event_count", "event_span_seconds", "is_upper_gap",
)


def candidates() -> dict[str, object]:
    common = dict(n_estimators=180, min_samples_leaf=5,
                  class_weight="balanced", random_state=42, n_jobs=-1)
    return {
        "ExtraTrees": make_pipeline(SimpleImputer(strategy="median"),
                                    ExtraTreesClassifier(max_depth=4, **common)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestClassifier(max_depth=4, **common)),
    }


def _fit(model, frame: pd.DataFrame, features: tuple[str, ...] = FEATURES) -> None:
    model.fit(frame[list(features)], frame.positive)


def _counts(frame: pd.DataFrame, selected: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(frame.positive, selected, labels=[0, 1]).ravel()
    return {
        "good_detected": int(tp), "good_missed": int(fn),
        "bad_rejected": int(tn), "bad_accepted": int(fp),
        "excellent_detected": int((frame.label.eq("Excelente").to_numpy() & selected).sum()),
        "balanced_accuracy": round(float(balanced_accuracy_score(frame.positive, selected)), 3),
    }


def _walk_forward(model, train: pd.DataFrame,
                  features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    days = sorted(train.session_date.unique())
    first = max(5, len(days) // 3)
    step = max(1, len(days) // 8)
    folds = []
    for start in range(first, len(days), step):
        earlier = train.loc[train.session_date.isin(days[:start])]
        later = train.loc[train.session_date.isin(days[start:start + step])].copy()
        if earlier.positive.nunique() < 2 or later.empty:
            continue
        fold_model = clone(model)
        _fit(fold_model, earlier, features)
        later["probability"] = fold_model.predict_proba(later[list(features)])[:, 1]
        folds.append(later)
    return pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()


def _choose_threshold(calibration: pd.DataFrame) -> tuple[float, float]:
    # Equilibrio de buenas y malas, sin imponer un ángulo o gap financiero.
    choices = []
    for threshold in np.arange(.20, .81, .02):
        selected = calibration.probability.to_numpy() >= threshold
        quality = balanced_accuracy_score(calibration.positive, selected)
        choices.append((quality, -abs(threshold - .5), float(threshold)))
    quality, _, threshold = max(choices)
    return threshold, float(quality)


def _three_label_check(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Diagnóstico separado: distinguir Excelente de Buena requiere más evidencia."""
    labels = ["Mala", "Buena", "Excelente"]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(n_estimators=180, max_depth=4, min_samples_leaf=5,
                             class_weight="balanced", random_state=42, n_jobs=-1))
    model.fit(train[list(FEATURES)], train.label)
    predicted = model.predict(test[list(FEATURES)])
    matrix = confusion_matrix(test.label, predicted, labels=labels)
    return {
        "labels": labels,
        "test_confusion_rows_actual_columns_predicted": matrix.tolist(),
        "test_macro_f1": round(float(f1_score(test.label, predicted, labels=labels,
                                              average="macro", zero_division=0)), 3),
        "test_correct_excellent": int(((test.label == "Excelente") &
                                        (predicted == "Excelente")).sum()),
        "test_total_excellent": int((test.label == "Excelente").sum()),
    }


def fit_trial_score_model(labeled_ten_seconds: pd.DataFrame):
    """Modelo exploratorio para Historial prueba; umbral sólo de pliegues previos.

    La vista se reajusta con todas las etiquetas guardadas. Sus filas etiquetadas
    son entrenamiento, mientras `study()` conserva la evaluación reservada.
    """
    training = labeled_ten_seconds.copy()
    training["is_upper_gap"] = training.target_band.eq("UPPER").astype(int)
    template = candidates()["ExtraTrees"]
    calibration = _walk_forward(template, training)
    if calibration.empty or calibration.positive.nunique() < 2:
        raise ValueError("Faltan fechas previas con ambas etiquetas para calibrar el score.")
    threshold, _ = _choose_threshold(calibration)
    model = clone(template)
    _fit(model, training)
    return model, threshold


def _support_grid(frame: pd.DataFrame) -> list[dict]:
    """Cruce observado de gap y ángulo; nunca sustituye el test cronológico."""
    observed = frame.loc[frame.gap_abs_pct.notna() &
                         frame.target_angle_deg_5m_equiv.notna()].copy()
    if len(observed) < 12:
        return []
    observed["gap_group"] = pd.qcut(observed.gap_abs_pct, 3,
                                    duplicates="drop").astype(str)
    observed["angle_group"] = pd.qcut(observed.target_angle_deg_5m_equiv, 3,
                                      duplicates="drop").astype(str)
    return [{"direction": direction, "gap_range_pct": gap, "angle_range_deg": angle,
             "cases": len(rows), "excellent": int(rows.label.eq("Excelente").sum()),
             "good": int(rows.label.eq("Buena").sum()),
             "bad": int(rows.label.eq("Mala").sum())}
            for (direction, gap, angle), rows in observed.groupby(
                ["target_band", "gap_group", "angle_group"], observed=True)]


def _contrast_pairs(test: pd.DataFrame) -> list[dict]:
    """Pares observados con gap parecido y etiquetas opuestas para revisar gráficas."""
    positives = test.loc[test.positive.eq(1) & test.gap_abs_pct.notna()]
    negatives = test.loc[test.positive.eq(0) & test.gap_abs_pct.notna()]
    pairs = []
    for good in positives.itertuples():
        for bad in negatives.itertuples():
            if good.target_band == bad.target_band:
                difference = abs(good.gap_abs_pct - bad.gap_abs_pct)
                pairs.append((difference, good, bad))
    pairs.sort(key=lambda item: item[0])
    selected, used = [], set()
    for difference, good, bad in pairs:
        if len(selected) >= 5:
            break
        good_key = (good.session_date, good.ticker)
        bad_key = (bad.session_date, bad.ticker)
        if good_key in used or bad_key in used:
            continue
        used.update((good_key, bad_key))
        selected.append({
            "gap_difference_pp": round(float(difference), 3),
            "good": {"date": good.session_date, "ticker": good.ticker,
                     "label": good.label, "gap_pct": round(float(good.gap_pct), 3),
                     "angle_deg": round(float(good.target_angle_deg_5m_equiv), 2)
                     if pd.notna(good.target_angle_deg_5m_equiv) else None,
                     "distance_pct": round(float(good.distance_outside_target_pct), 3),
                     "prior_width_pct": round(float(good.prior_width_pct), 3),
                     "band_reaction_pct": round(float(good.target_band_reaction_pct), 3),
                     "model_score": getattr(good, "ExtraTrees_score_raw_0_100", None)},
            "bad": {"date": bad.session_date, "ticker": bad.ticker,
                    "label": bad.label, "gap_pct": round(float(bad.gap_pct), 3),
                    "angle_deg": round(float(bad.target_angle_deg_5m_equiv), 2)
                    if pd.notna(bad.target_angle_deg_5m_equiv) else None,
                    "distance_pct": round(float(bad.distance_outside_target_pct), 3),
                    "prior_width_pct": round(float(bad.prior_width_pct), 3),
                    "band_reaction_pct": round(float(bad.target_band_reaction_pct), 3),
                    "model_score": getattr(bad, "ExtraTrees_score_raw_0_100", None)},
        })
    return selected


def study() -> tuple[dict, pd.DataFrame]:
    labels, examples = labeled_examples()
    report = {
        "generated_at_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "objective": "Excelente/Buena frente a Mala; prueba separada de tres etiquetas",
        "limitations": ["Etiquetas elegidas manualmente, no muestra aleatoria del universo.",
                        "Las horas SIP históricas no prueban latencia de recepción en vivo.",
                        "El porcentaje del modelo no es una probabilidad calibrada."],
        "features": list(FEATURES),
        "financial_angle_or_gap_cutoff_applied": False,
        "total_manual_labels": labels.label.value_counts().to_dict(),
        "windows": {},
    }
    predictions = []
    ten = examples.loc[examples.window_seconds.eq(10)]
    days = sorted(ten.session_date.unique())
    if len(days) < 8:
        raise ValueError("No hay suficientes fechas con cortes reales de 10 s.")
    boundary = days[int(len(days) * .75)]
    report["heldout_from_date"] = boundary
    for seconds in WINDOWS:
        subset = examples.loc[examples.window_seconds.eq(seconds)].copy()
        window = {"cases": len(subset), "days": int(subset.session_date.nunique()),
                  "labels": subset.label.value_counts().to_dict(),
                  "source_counts": subset.source.value_counts().to_dict()}
        report["windows"][str(seconds)] = window
        train = subset.loc[subset.session_date < boundary].copy()
        test = subset.loc[subset.session_date >= boundary].copy()
        window["training_cases"] = len(train)
        window["test_cases"] = len(test)
        if (train.session_date.nunique() < 8 or len(test) < 15 or
                train.positive.nunique() < 2 or test.positive.nunique() < 2):
            window["status"] = "sin_datos_suficientes_para_validacion_temporal"
            continue
        train["is_upper_gap"] = train.target_band.eq("UPPER").astype(int)
        test["is_upper_gap"] = test.target_band.eq("UPPER").astype(int)
        window["baseline_score_70"] = _counts(test, test.existing_score_at_cutoff.ge(70).to_numpy())
        window["observed_gap_angle_grid_training"] = _support_grid(train)
        trials = {}
        for name, template in candidates().items():
            calibration = _walk_forward(template, train)
            if calibration.empty or calibration.positive.nunique() < 2:
                continue
            threshold, oof_quality = _choose_threshold(calibration)
            model = clone(template)
            _fit(model, train)
            probability = model.predict_proba(test[list(FEATURES)])[:, 1]
            selected = probability >= threshold
            trials[name] = {"threshold_from_training": round(threshold, 3),
                            "training_oof_balanced_accuracy": round(oof_quality, 3),
                            "training_oof_cases": len(calibration),
                            "test": _counts(test, selected),
                            "test_roc_auc": round(float(roc_auc_score(test.positive, probability)), 3),
                            "feature_importance_training": dict(zip(
                                FEATURES, model.steps[-1][1].feature_importances_.round(3).tolist()))}
            test[f"{name}_score_raw_0_100"] = (probability * 100).round(1)
            test[f"{name}_selected"] = selected
        if not trials:
            window["status"] = "sin_pliegues_temporales_utiles"
            continue
        window["similar_gap_opposite_labels_test"] = _contrast_pairs(test)
        # Se elige con fechas de entrenamiento, jamás con resultados del test.
        winner = max(trials, key=lambda name: trials[name]["training_oof_balanced_accuracy"])
        ablations = {}
        for description, used_features in {
                "sólo_gap_y_distancia": ("gap_pct", "gap_abs_pct",
                                         "distance_outside_target_pct", "is_upper_gap"),
                "todos_menos_ángulo": tuple(feature for feature in FEATURES
                                            if feature != "target_angle_deg_5m_equiv"),
        }.items():
            template = candidates()["ExtraTrees"]
            calibration = _walk_forward(template, train, used_features)
            if calibration.empty or calibration.positive.nunique() < 2:
                continue
            threshold, oof_quality = _choose_threshold(calibration)
            model = clone(template)
            _fit(model, train, used_features)
            probability = model.predict_proba(test[list(used_features)])[:, 1]
            ablations[description] = {
                "training_oof_balanced_accuracy": round(oof_quality, 3),
                "threshold_from_training": round(threshold, 3),
                "test": _counts(test, probability >= threshold),
                "test_roc_auc": round(float(roc_auc_score(test.positive, probability)), 3),
            }
        window.update(status="validated_on_later_dates", selected_model=winner,
                      models=trials, ablations_exploratory=ablations,
                      three_label_exploratory=_three_label_check(train, test),
                      train_dates=[train.session_date.min(), train.session_date.max()],
                      test_dates=[test.session_date.min(), test.session_date.max()])
        keep = ["session_date", "ticker", "window_seconds", "label", "source",
                "existing_score_at_cutoff", *FEATURES,
                *[name for name in test.columns if name.endswith(("_score_raw_0_100", "_selected"))]]
        predictions.append(test[keep])
    return report, pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def update_universe_audit(report: dict, stamp: str) -> Path:
    """Refresh the exploratory universe view without changing held-out metrics."""
    from evaluate_manual_labels import build_trial_history
    _, examples = labeled_examples()
    universe = build_trial_history(examples, mode="ai")
    universe_path = OUTPUT_DIR / f"early_ai_universe_{stamp}.csv"
    universe.to_csv(universe_path, index=False)
    observed = universe.loc[universe.early_data_status.eq("Disponible")]
    report["universe_audit"] = {
        "observed_cuts": len(observed),
        "ai_score_ge70": int(observed.experimental_score_10s.ge(70).sum()),
        "outside_prior_band_and_ai_score_ge70": int(observed.trial_candidate.sum()),
        "unlabeled_ai_score_ge70": int((observed.manual_label.eq("Pendiente") &
                                        observed.experimental_score_10s.ge(70)).sum()),
        "warning": "La mayoría del universo no fue etiquetada; un score alto no "
                   "valida esos casos. Por eso este modelo no sustituye Historial prueba.",
    }
    return universe_path


def main(audit_only: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    report_path = OUTPUT_DIR / f"early_ai_study_{stamp}.json"
    prediction_path = OUTPUT_DIR / f"early_ai_predictions_{stamp}.csv"
    if audit_only:
        if not report_path.exists():
            raise ValueError("Primero ejecuta el estudio completo de IA.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report, predictions = study()
    # La auditoría no participa en la elección del modelo ni en las métricas reservadas.
    universe_path = update_universe_audit(report, stamp)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if audit_only:
        print(f"Auditoría actualizada: {universe_path}")
        return
    predictions.to_csv(prediction_path, index=False)
    print(json.dumps({"windows": {key: {item: value for item, value in row.items()
                                      if item not in ("observed_gap_angle_grid_training",
                                                      "similar_gap_opposite_labels_test")}
                                   for key, row in report["windows"].items()}},
                     ensure_ascii=False, indent=2))
    print(f"Reporte: {report_path}\nPredicciones reservadas: {prediction_path}"
          f"\nAuditoría de las 50 sesiones: {universe_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    main(audit_only=parser.parse_args().audit_only)
