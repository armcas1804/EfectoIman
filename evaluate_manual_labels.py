"""Evalúa juicios visuales con información disponible a 10, 20 o 30 segundos.

Las etiquetas son juicios humanos posteriores; las variables se construyen sólo
con velas previas y operaciones hasta cada corte. No modifica el scanner.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR
from daily_store import export_history
from manual_review import early_features, early_observations, load_reviews
from opening_scan import _score
from scanner_settings import DEFAULT_THRESHOLDS, MAX_UPPER_ANGLE_DEG


FEATURES = (
    "gap_abs_pct",
    "distance_outside_target_pct",
    "prior_width_pct",
    "prior_width_change_pp_per_bar",
    "target_angle_deg_5m_equiv",
    "existing_score_at_cutoff",
)
WINDOWS = (10, 20, 30)
OUTPUT_DIR = DATA_DIR / "snapshots"


def trial_signal_eligible(row: pd.Series) -> bool:
    """Hay gap suficiente y precio fuera de la banda al corte de 10 s."""
    gap = row.get("gap_pct")
    distance = row.get("distance_outside_target_pct")
    minimum = DEFAULT_THRESHOLDS.minimum_gap_pct
    return bool(pd.notna(gap) and pd.notna(distance) and distance > 0 and
                (minimum is None or abs(gap) >= minimum))


def ai_opening_eligible(row: pd.Series) -> bool:
    """Condición esencial: al corte el precio está fuera de la banda previa."""
    distance = row.get("distance_outside_prior_target_pct")
    gap = row.get("gap_pct")
    opening_direction = row.get("opening_gap_direction")
    expected_band = {"UP": "UPPER", "DOWN": "LOWER"}.get(opening_direction)
    if expected_band is not None and row.get("target_band") != expected_band:
        return False
    return bool(pd.notna(distance) and pd.notna(gap) and distance > 0 and gap != 0)


def visible_trial_score(row: pd.Series) -> float:
    """Conserva el score real de una señal; no crea un 69 artificial."""
    return (float(row.model_score_without_filters_10s)
            if trial_signal_eligible(row) and
            pd.notna(row.get("model_score_without_filters_10s")) else np.nan)


def trial_geometry_at_10s(row: pd.Series) -> bool:
    """Reglas geométricas vigentes del corte, sin usar el score."""
    return not trial_geometry_failures(row)


def trial_geometry_failures(row: pd.Series) -> list[str]:
    """Nombra las condiciones que impiden entrar en la lista estricta."""
    required = ("gap_pct", "distance_outside_target_pct", "prior_width_pct",
                "prior_width_change_pp_per_bar", "target_angle_deg_5m_equiv")
    if any(pd.isna(row.get(name)) for name in required):
        return ["Faltan métricas del corte"]
    limits = DEFAULT_THRESHOLDS
    failures = []
    if row.distance_outside_target_pct <= 0:
        failures.append("Dentro de Bollinger")
    if (limits.minimum_gap_pct is not None and
            abs(row.gap_pct) < limits.minimum_gap_pct):
        failures.append("Gap insuficiente")
    if (limits.minimum_distance_to_band_pct is not None and
            row.distance_outside_target_pct < limits.minimum_distance_to_band_pct):
        failures.append("Distancia exterior insuficiente")
    if (limits.maximum_prior_bb_width_pct is not None and
            row.prior_width_pct > limits.maximum_prior_bb_width_pct):
        failures.append("Banda previa ancha")
    if (limits.maximum_prior_bb_width_growth is not None and
            row.prior_width_change_pp_per_bar > limits.maximum_prior_bb_width_growth):
        failures.append("Banda previa expandiéndose")
    if row.target_angle_deg_5m_equiv > MAX_UPPER_ANGLE_DEG:
        failures.append("Ángulo sobre el límite")
    return failures


def trial_candidate_at_10s(row: pd.Series) -> bool:
    """Candidato cuando cumple la geometría y el score de prueba alcanza 70."""
    return bool(pd.notna(row.get("experimental_score_10s")) and
                row.experimental_score_10s >= 70 and trial_geometry_at_10s(row))


def score_at_cutoff(row: pd.Series) -> float:
    """Reconstruye el score operativo con métricas conocidas al corte."""
    return _score({
        "gap_pct": row.gap_pct,
        "distance_to_band_pct": row.distance_outside_target_pct,
        "prior_bb_width_pct": row.prior_width_pct,
        "prior_bb_width_change": row.prior_width_change_pp_per_bar,
        "target_angle_cutoff_deg": row.target_angle_deg_5m_equiv,
        "target_angle_10s_deg": row.target_angle_deg_5m_equiv,
        "cutoff_bb_upper": row.get("upper_provisional", np.nan),
        "cutoff_bb_lower": row.get("lower_provisional", np.nan),
    }, "open")


def labeled_examples() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = load_reviews()
    setups, candles = export_history(100)
    reviews = reviews.loc[reviews.label.isin(("Excelente", "Buena", "Mala"))]
    if reviews.empty or setups.empty or candles.empty:
        return pd.DataFrame(), pd.DataFrame()
    joined = reviews.merge(setups[["session_date", "ticker", "score"]],
                           on=["session_date", "ticker"], how="left", validate="one_to_one")
    joined["positive"] = joined.label.isin(("Excelente", "Buena")).astype(int)
    records = []
    for day, labeled_day in joined.groupby("session_date"):
        observed = early_observations(day)
        if observed.empty:
            continue
        day_candles = candles.loc[candles.session_date.eq(day)]
        for row in labeled_day.itertuples():
            features = early_features(day, row.ticker, day_candles, observed)
            if features.empty:
                continue
            features["label"] = row.label
            features["positive"] = row.positive
            features["existing_final_score"] = row.score
            records.append(features)
    examples = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if not examples.empty:
        examples["gap_abs_pct"] = examples.gap_pct.abs()
        # Misma fórmula operativa, recalculada con el precio y ángulo del corte.
        examples["existing_score_at_cutoff"] = examples.apply(score_at_cutoff, axis=1)
    return joined, examples


def chronological_split(frame: pd.DataFrame, test_fraction: float = .25):
    days = sorted(frame.session_date.unique())
    if len(days) < 8:
        raise ValueError("Se necesitan al menos ocho fechas con etiquetas y operaciones.")
    boundary = days[max(1, int(len(days) * (1 - test_fraction)))]
    train = frame.loc[frame.session_date < boundary].copy()
    test = frame.loc[frame.session_date >= boundary].copy()
    if train.positive.nunique() != 2:
        raise ValueError("El período de entrenamiento requiere etiquetas de ambas clases.")
    return train, test


def counts(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positive_recall = tp / (tp + fn) if tp + fn else None
    negative_recall = tn / (tn + fp) if tn + fp else None
    return {
        "tp_buenas": int(tp), "fn_buenas": int(fn),
        "tn_malas": int(tn), "fp_malas": int(fp),
        "precision_buenas": round(float(tp / (tp + fp)), 3) if tp + fp else None,
        "recall_buenas": round(float(positive_recall), 3) if positive_recall is not None else None,
        "recall_malas": round(float(negative_recall), 3) if negative_recall is not None else None,
        "exactitud_equilibrada": round(float((positive_recall + negative_recall) / 2), 3)
        if positive_recall is not None and negative_recall is not None else None,
    }


def choose_training_threshold(labels, scores: np.ndarray) -> float:
    """Favorece rechazar malas, conservando al menos 60% de buenas observadas."""
    choices = []
    for threshold in np.linspace(.01, .99, 99):
        outcome = counts(labels, scores >= threshold)
        if (outcome["recall_buenas"] is not None and
                outcome["recall_malas"] is not None and
                outcome["recall_buenas"] >= .6):
            # La tasa de malas rechazadas pesa dos veces la de buenas detectadas.
            utility = (outcome["recall_buenas"] + 2 * outcome["recall_malas"]) / 3
            choices.append((utility, threshold))
    return float(max(choices)[1]) if choices else .5


def experimental_score(raw_probability: np.ndarray, threshold: float) -> np.ndarray:
    """Mapeo monótono 0→0, umbral→70, 1→100; no es probabilidad calibrada."""
    scores = np.asarray(raw_probability, dtype=float)
    return np.where(scores < threshold,
                    70 * scores / threshold,
                    70 + 30 * (scores - threshold) / (1 - threshold))


def _new_model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=False),
                         RandomForestClassifier(n_estimators=150, max_depth=3,
                                                min_samples_leaf=5,
                                                class_weight="balanced",
                                                random_state=42))


def chronological_training_predictions(train: pd.DataFrame) -> pd.DataFrame:
    """Predicciones sobre fechas posteriores dentro del entrenamiento únicamente."""
    days = sorted(train.session_date.unique())
    first = max(4, len(days) // 3)
    step = max(1, len(days) // 7)
    predictions = []
    for stop in range(first, len(days), step):
        earlier = train.loc[train.session_date.isin(days[:stop])]
        later = train.loc[train.session_date.isin(days[stop:stop + step])].copy()
        if earlier.positive.nunique() < 2 or later.empty:
            continue
        model = _new_model()
        model.fit(earlier[list(FEATURES)], earlier.positive,
                  randomforestclassifier__sample_weight=np.where(
                      earlier.label.eq("Excelente"), 2.0, 1.0))
        later["training_probability"] = model.predict_proba(
            later[list(FEATURES)])[:, 1]
        predictions.append(later)
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def fit_model(train: pd.DataFrame):
    calibration = chronological_training_predictions(train)
    threshold = (choose_training_threshold(calibration.positive,
                                           calibration.training_probability.to_numpy())
                 if len(calibration) >= 12 and calibration.positive.nunique() == 2
                 else .5)
    evidence = (counts(calibration.positive,
                       calibration.training_probability.ge(threshold))
                if not calibration.empty else {})
    model = _new_model()
    model.fit(train[list(FEATURES)], train.positive,
              randomforestclassifier__sample_weight=np.where(train.label.eq("Excelente"), 2.0, 1.0))
    return model, threshold, {"rows": len(calibration),
                              "dates": int(calibration.session_date.nunique())
                              if not calibration.empty else 0,
                              "metrics": evidence}


def evaluate_window(frame: pd.DataFrame, seconds: int) -> tuple[dict, pd.DataFrame]:
    subset = frame.loc[frame.window_seconds.eq(seconds)].copy()
    result = {"window_seconds": seconds, "examples": len(subset),
              "dates": int(subset.session_date.nunique()),
              "labels": subset.label.value_counts().to_dict()}
    if len(subset) < 30 or subset.session_date.nunique() < 8 or subset.positive.nunique() < 2:
        result["status"] = "datos_insuficientes_para_entrenar_y_validar"
        return result, pd.DataFrame()
    train, test = chronological_split(subset)
    result.update({"status": "evaluado", "train_dates": [train.session_date.min(), train.session_date.max()],
                   "test_dates": [test.session_date.min(), test.session_date.max()],
                   "train_rows": len(train), "test_rows": len(test),
                   "train_labels": train.label.value_counts().to_dict(),
                   "test_labels": test.label.value_counts().to_dict(),
                   "features": list(FEATURES)})
    # Imputador, árboles y pesos se ajustan exclusivamente en las fechas de entrenamiento.
    model, threshold, calibration = fit_model(train)
    test["model_raw_score"] = model.predict_proba(test[list(FEATURES)])[:, 1] * 100
    # Score experimental: el corte seleccionado en entrenamiento se muestra como 70.
    # No es una probabilidad de éxito ni modifica el score operativo.
    test["model_score_experimental"] = experimental_score(
        test.model_raw_score.to_numpy() / 100, threshold)
    test["model_ge_70"] = test.model_raw_score.ge(100 * threshold)
    if seconds == 10:
        test["experimental_score_10s"] = test.model_score_experimental
        test["trial_candidate"] = test.apply(trial_candidate_at_10s, axis=1)
        result["trial_candidate_with_existing_filters"] = counts(
            test.positive, test.trial_candidate)
    result["training_threshold_raw_score"] = round(100 * threshold, 1)
    result["training_chronological_calibration"] = calibration
    result["threshold_selection"] = (
        "Cortes sucesivos por fecha dentro del entrenamiento; recall de malas con peso doble "
        "y recall de buenas >= 60%. Las fechas de prueba final no eligen el umbral.")
    result["model_ge_70"] = counts(test.positive, test.model_ge_70)
    result["existing_score_at_cutoff_ge_70"] = counts(
        test.positive, test.existing_score_at_cutoff.ge(70))
    result["feature_medians_by_label"] = {
        label: {feature: (round(float(value), 3) if pd.notna(value) else None)
                for feature, value in medians.items()}
        for label, medians in subset.groupby("label")[list(FEATURES)].median().iterrows()
    }
    result["feature_importances_exploratory"] = {
        feature: round(float(importance), 3)
        for feature, importance in zip(
            FEATURES, model.named_steps["randomforestclassifier"].feature_importances_)
    }
    result["importance_warning"] = (
        "Importancias de árboles sobre casos seleccionados para revisión; "
        "no son causalidad ni pesos operativos aprobados.")
    result["always_good"] = counts(test.positive, np.ones(len(test), dtype=int))
    if test.positive.nunique() == 2:
        result["average_precision"] = round(float(average_precision_score(
            test.positive, test.model_raw_score)), 3)
        result["roc_auc"] = round(float(roc_auc_score(
            test.positive, test.model_raw_score)), 3)
    else:
        result["warning"] = "La prueba sólo contiene una clase; no mide separación entre buenas y malas."
    return result, test


def build_trial_history(labeled: pd.DataFrame, mode: str = "legacy") -> pd.DataFrame:
    """Score exploratorio previo o auditoría IA, con validación externa separada."""
    ten_second_labels = labeled.loc[labeled.window_seconds.eq(10)]
    if mode == "ai":
        from study_early_ai import FEATURES as AI_FEATURES, fit_trial_score_model
        model, threshold = fit_trial_score_model(ten_second_labels)
    elif mode == "legacy":
        model, threshold, _ = fit_model(ten_second_labels)
    else:
        raise ValueError(f"Modo de historial desconocido: {mode}")
    training_keys = set(zip(ten_second_labels.session_date, ten_second_labels.ticker))
    setups, candles = export_history(100)
    reviews = load_reviews()
    labels = {(row.session_date, row.ticker): row.label for row in reviews.itertuples()}
    records = []
    historical_days = {path.stem.removeprefix("apertura_historica_10s_") for path in
                       sorted(OUTPUT_DIR.glob("apertura_historica_10s_*.csv"))[-50:]}
    for day, day_setups in setups.loc[setups.session_date.isin(historical_days)].groupby(
            "session_date"):
        observations = early_observations(day)
        available = set(observations.loc[observations.window_seconds.eq(10), "ticker"])
        day_candles = candles.loc[candles.session_date.eq(day)]
        for row in day_setups.itertuples():
            record = {"session_date": day, "ticker": row.ticker,
                      "source": "SIN_CORTE", "early_data_status": "Sin 10 s"}
            if row.ticker in available:
                features = early_features(day, row.ticker, day_candles, observations)
                if not features.empty:
                    first = features.loc[features.window_seconds.eq(10)]
                    if not first.empty:
                        record.update(first.iloc[0].to_dict())
                        record["early_data_status"] = "Disponible"
            record.update(opening_gap_direction=row.gap_direction,
                          existing_final_score=row.score,
                          existing_final_distance_pct=row.distance_to_band_pct,
                          manual_label=labels.get((day, row.ticker), "Pendiente"),
                          trial_model_scope=("Caso etiquetado usado para entrenar"
                                             if (day, row.ticker) in training_keys else
                                             "Sin juicio para entrenamiento"))
            records.append(record)
    history = pd.DataFrame.from_records(records)
    if history.empty:
        return history
    history["gap_abs_pct"] = history.gap_pct.abs()
    history["is_upper_gap"] = history.target_band.eq("UPPER").astype(int)
    history["existing_score_at_cutoff"] = np.nan
    history["experimental_score_10s"] = np.nan
    history["model_score_without_filters_10s"] = np.nan
    available = history.early_data_status.eq("Disponible")
    history.loc[available, "existing_score_at_cutoff"] = history.loc[available].apply(
        score_at_cutoff, axis=1)
    if mode == "ai":
        raw = model.predict_proba(history.loc[available, list(AI_FEATURES)])[:, 1]
        history.loc[available, "model_score_without_filters_10s"] = (raw * 100).round(1)
        history.loc[available, "experimental_score_10s"] = (
            experimental_score(raw, threshold).clip(1, 100).round(1))
        signal = history.apply(ai_opening_eligible, axis=1) & available
        direction_match = history.opening_gap_direction.map(
            {"UP": "UPPER", "DOWN": "LOWER"}).eq(history.target_band)
        history["trial_failed_conditions_10s"] = np.select(
            [~available, ~direction_match, ~signal,
             history.experimental_score_10s.lt(70)],
            ["Sin datos de 10 s", "Cambió dirección del gap",
             "Precio dentro de banda previa", "Score IA menor que 70"],
            default="")
        history["trial_signal_status"] = np.select(
            [~available, ~direction_match, ~signal,
             history.experimental_score_10s.lt(70)],
            ["Sin datos de 10 s", "Dirección cambió", "Sin apertura exterior",
             "Score IA menor que 70"],
            default="Candidato IA")
        history["trial_candidate"] = signal & history.experimental_score_10s.ge(70)
    else:
        raw = model.predict_proba(history.loc[available, list(FEATURES)])[:, 1]
        history.loc[available, "model_score_without_filters_10s"] = (
            experimental_score(raw, threshold).round(1))
        signal = history.apply(trial_signal_eligible, axis=1) & available
        history["trial_failed_conditions_10s"] = history.apply(
            lambda row: ", ".join(trial_geometry_failures(row)), axis=1)
        geometry = history.trial_failed_conditions_10s.eq("")
        history.loc[signal, "experimental_score_10s"] = history.loc[signal].apply(
            visible_trial_score, axis=1)
        history["trial_signal_status"] = np.select(
            [~available, history.distance_outside_target_pct.le(0),
             available & ~signal, signal & ~geometry],
            ["Sin datos de 10 s", "Dentro de Bollinger", "Gap insuficiente",
             "No cumple filtros del scanner"], default="Señal evaluable")
        history["trial_candidate"] = history.apply(trial_candidate_at_10s, axis=1)
        history.loc[history.trial_candidate, "trial_signal_status"] = "Candidato"
        history.loc[signal & geometry & ~history.trial_candidate,
                    "trial_signal_status"] = "Score menor que 70"
    history["score_difference_10s"] = (
        history.experimental_score_10s - history.existing_score_at_cutoff).round(1)
    history["evidence_scope"] = np.where(
        history.existing_final_score.ge(70) | history.manual_label.ne("Pendiente"),
        "Comparable al grupo etiquetado", "Fuera del grupo etiquetado")
    history["low_score_context"] = np.select(
        [history.manual_label.ne("Pendiente"), history.existing_final_score.lt(40),
         history.existing_final_score.lt(60)],
        ["Juicio manual", "Score final <40: hipótesis Mala",
         "Score final 40-59: revisar"], default="Sin juicio")
    return history.sort_values(["session_date", "experimental_score_10s", "ticker"],
                               ascending=[False, False, True]).reset_index(drop=True)


def summarize_trial_history(history: pd.DataFrame) -> pd.DataFrame:
    """Cobertura y selección por fecha; sin asumir que Pendiente significa Mala."""
    records = []
    for day, rows in history.groupby("session_date", sort=True):
        observed = rows.loc[rows.early_data_status.eq("Disponible")]
        old_high = observed.existing_score_at_cutoff.ge(70)
        new_high = observed.experimental_score_10s.ge(70)
        records.append({
            "session_date": day,
            "monitored": len(rows), "with_10s": len(observed),
            "old_score_ge70": int(old_high.sum()),
            "trial_score_ge70": int(new_high.sum()),
            "trial_candidates": int(observed.trial_candidate.sum()),
            "scores_up": int((observed.score_difference_10s > 0).sum()),
            "scores_down": int((observed.score_difference_10s < 0).sum()),
            "unreviewed_low40_trial_ge70": int((
                observed.manual_label.eq("Pendiente") &
                observed.existing_final_score.lt(40) & new_high).sum()),
            "manual_good_high": int((
                observed.manual_label.isin(("Excelente", "Buena")) & new_high).sum()),
            "manual_bad_high": int((observed.manual_label.eq("Mala") & new_high).sum()),
            "manual_good_candidates": int((
                observed.manual_label.isin(("Excelente", "Buena")) &
                observed.trial_candidate).sum()),
            "manual_bad_candidates": int((
                observed.manual_label.eq("Mala") & observed.trial_candidate).sum()),
            "unreviewed_candidates": int((
                observed.manual_label.eq("Pendiente") & observed.trial_candidate).sum()),
        })
    return pd.DataFrame.from_records(records).sort_values("session_date", ascending=False)


def evaluate_weak_low_score_context(labeled: pd.DataFrame,
                                    history: pd.DataFrame) -> dict:
    """Prueba separada de pseudo negativos; nunca se incorporan al score principal."""
    train, test = chronological_split(labeled.loc[labeled.window_seconds.eq(10)])
    weak = history.loc[
        history.session_date.lt(test.session_date.min()) &
        history.manual_label.eq("Pendiente") &
        history.existing_final_score.lt(40) &
        history.existing_final_distance_pct.gt(0) &
        history.early_data_status.eq("Disponible")
    ].copy()
    if weak.empty:
        return {"status": "sin_casos_de_contexto"}
    weak["positive"] = 0
    combined = pd.concat([train, weak], ignore_index=True)
    weights = np.r_[np.where(train.label.eq("Excelente"), 1.5, 1.0),
                    np.full(len(weak), .2)]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced", C=.25))
    model.fit(combined[list(FEATURES)], combined.positive,
              logisticregression__sample_weight=weights)
    threshold = choose_training_threshold(
        train.positive, model.predict_proba(train[list(FEATURES)])[:, 1])
    predicted = model.predict_proba(test[list(FEATURES)])[:, 1] >= threshold
    return {"status": "evaluado", "hypothetical_negative_examples": len(weak),
            "pseudo_label_weight": .2, "threshold_raw_score": round(100 * threshold, 1),
            "test_manual_labels_only": counts(test.positive, predicted),
            "warning": "Pseudo etiquetas basadas en el score final; no son juicios humanos."}


def main() -> None:
    labels, examples = labeled_examples()
    if labels.empty:
        raise SystemExit("No hay juicios visuales ni diagnósticos locales.")
    report = {
        "generated_at_et": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "target": "Excelente/Buena=1; Mala=0; Dudosa excluida",
        "manual_labels": labels.label.value_counts().to_dict(),
        "labeled_dates": int(labels.session_date.nunique()),
        "selection_warning": (
            "Los juicios humanos provienen de casos seleccionados para revisión; "
            "la validación sólo mide separación en ese grupo y no valida el universo "
            "completo ni la lista final de candidatos."),
        "final_score_ge_70_vs_manual_labels": counts(labels.positive, labels.score.ge(70)),
        "final_score_warning": "Este score puede incorporar datos posteriores; sólo diagnostica el score actual.",
        "windows": {},
    }
    predictions = []
    for seconds in WINDOWS:
        if examples.empty:
            window_report = {"window_seconds": seconds, "examples": 0,
                             "status": "datos_insuficientes_para_entrenar_y_validar"}
            prediction = pd.DataFrame()
        else:
            window_report, prediction = evaluate_window(examples, seconds)
        report["windows"][str(seconds)] = window_report
        if not prediction.empty:
            predictions.append(prediction)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    report_path = OUTPUT_DIR / f"manual_label_validation_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            OUTPUT_DIR / f"manual_label_predictions_{stamp}.csv", index=False)
    if report["windows"]["10"].get("status") == "evaluado":
        comparison = build_trial_history(examples)
        if not comparison.empty:
            report["history_trial_model"] = (
                "Bosque exploratorio previo, reajustado con todas las etiquetas "
                "humanas que tienen corte real de 10 s. "
                "Los scores de casos etiquetados en Historial prueba son de entrenamiento; "
                "la evaluación fuera de muestra está en windows/10/model_ge_70.")
            comparison.to_csv(OUTPUT_DIR / f"manual_label_history_trial_{stamp}.csv",
                              index=False)
            summary = summarize_trial_history(comparison)
            summary.to_csv(OUTPUT_DIR / f"manual_label_daily_summary_{stamp}.csv", index=False)
            report["history_trial_rows"] = len(comparison)
            report["history_trial_days"] = len(summary)
            report["history_trial_scored"] = int(comparison.early_data_status.eq("Disponible").sum())
            report["history_trial_score_ge70"] = int(comparison.experimental_score_10s.ge(70).sum())
            report["history_trial_candidates"] = int(comparison.trial_candidate.sum())
            report["weak_low_score_context_experiment"] = (
                evaluate_weak_low_score_context(examples, comparison))
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
