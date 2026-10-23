"""Compara clasificadores de juicio visual con datos disponibles hasta cada corte.

La elección de variables, modelo y umbral utiliza sólo fechas anteriores al
período de evaluación. Los juicios manuales no son una muestra aleatoria del
universo y esta evaluación retrospectiva no demuestra rentabilidad.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR
from daily_store import export_history
from evaluate_manual_labels import experimental_score, labeled_examples, score_at_cutoff
from manual_review import early_features, early_observations, load_reviews


OUTPUT_DIR = DATA_DIR / "snapshots"
BASE_FEATURES = (
    "gap_abs_pct", "distance_outside_target_pct", "prior_width_pct",
    "prior_width_change_pp_per_bar", "prior_target_slope_pct_per_bar",
    "target_angle_deg_5m_equiv", "target_band_reaction_pct", "is_upper_gap",
)
WIDTH_FEATURES = BASE_FEATURES + ("provisional_width_ratio",)
TRAJECTORY_FEATURES = WIDTH_FEATURES + (
    "gap_abs_pct_10s", "distance_outside_target_pct_10s",
    "target_angle_deg_5m_equiv_10s", "provisional_width_ratio_10s",
    "price_move_10_to_cutoff_pct", "angle_change_10_to_cutoff_deg",
)
OPEN_GEOMETRY_FEATURES = TRAJECTORY_FEATURES + (
    "distance_outside_prior_target_pct", "z_at_cutoff",
    "prior_width_growth_pct_per_bar", "log_event_count",
)
SCORE_CORRECTION_FEATURES = (
    "existing_score_at_cutoff", "gap_abs_pct",
    "distance_outside_target_pct", "target_angle_deg_5m_equiv",
    "prior_width_pct", "provisional_width_ratio",
    "prior_width_change_pp_per_bar", "is_upper_gap",
)


def feature_table(examples: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Crea variables causales; no une ningún corte posterior a `seconds`."""
    working = examples.copy()
    if "gap_abs_pct" not in working:
        working["gap_abs_pct"] = working.gap_pct.abs()
    current = working.loc[working.window_seconds.eq(seconds)].copy()
    if current.empty:
        return current
    current["is_upper_gap"] = current.target_band.eq("UPPER").astype(int)
    current["provisional_width_ratio"] = (
        (current.upper_provisional - current.lower_provisional) /
        ((current.upper_provisional + current.lower_provisional) / 2) /
        (current.prior_width_pct / 100))
    current["log_event_count"] = np.log1p(current.event_count.clip(lower=0))
    if seconds > 10:
        prior = working.loc[working.window_seconds.eq(10)].copy()
        prior["provisional_width_ratio_10s"] = (
            (prior.upper_provisional - prior.lower_provisional) /
            ((prior.upper_provisional + prior.lower_provisional) / 2) /
            (prior.prior_width_pct / 100))
        prior = prior[["session_date", "ticker", "price", "gap_abs_pct",
                       "distance_outside_target_pct", "target_angle_deg_5m_equiv",
                       "provisional_width_ratio_10s"]].rename(columns={
                           "price": "price_10s", "gap_abs_pct": "gap_abs_pct_10s",
                           "distance_outside_target_pct": "distance_outside_target_pct_10s",
                           "target_angle_deg_5m_equiv": "target_angle_deg_5m_equiv_10s"})
        current = current.merge(prior, on=["session_date", "ticker"], how="inner",
                                validate="one_to_one")
        direction = np.where(current.is_upper_gap.eq(1), 1, -1)
        current["price_move_10_to_cutoff_pct"] = direction * (
            current.price / current.price_10s - 1) * 100
        current["angle_change_10_to_cutoff_deg"] = (
            current.target_angle_deg_5m_equiv -
            current.target_angle_deg_5m_equiv_10s)
    return current.sort_values(["session_date", "ticker"]).reset_index(drop=True)


def candidates() -> dict[str, object]:
    tree = dict(n_estimators=160, min_samples_leaf=6,
                class_weight="balanced", random_state=42, n_jobs=-1)
    return {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=.3, class_weight="balanced",
                                                     max_iter=2000)),
        "extra_trees": make_pipeline(SimpleImputer(strategy="median"),
                                     ExtraTreesClassifier(max_depth=4, **tree)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(max_depth=4, **tree)),
        "gradient_boosting": make_pipeline(SimpleImputer(strategy="median"),
                                           GradientBoostingClassifier(
                                               n_estimators=80, learning_rate=.04,
                                               max_depth=2, min_samples_leaf=7,
                                               random_state=42)),
        "hist_gradient": make_pipeline(SimpleImputer(strategy="median"),
                                       HistGradientBoostingClassifier(
                                           max_iter=80, max_leaf_nodes=7,
                                           min_samples_leaf=8, l2_regularization=5,
                                           random_state=42)),
    }


def counts(y: pd.Series, selected: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, selected, labels=[0, 1]).ravel()
    return {"good_detected": int(tp), "good_missed": int(fn),
            "bad_rejected": int(tn), "bad_accepted": int(fp),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, selected)), 3),
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0}


def inner_predictions(model, train: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    days = sorted(train.session_date.unique())
    first = max(10, len(days) // 3)
    step = max(2, (len(days) - first) // 5)
    folds = []
    for stop in range(first, len(days), step):
        earlier = train.loc[train.session_date.isin(days[:stop])]
        later = train.loc[train.session_date.isin(days[stop:stop + step])].copy()
        if earlier.positive.nunique() < 2 or later.empty:
            continue
        fitted = clone(model)
        fitted.fit(earlier[list(features)], earlier.positive)
        later["probability"] = fitted.predict_proba(later[list(features)])[:, 1]
        folds.append(later)
    return pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()


def select_threshold(calibration: pd.DataFrame) -> tuple[float, dict]:
    """Ajusta el umbral dentro del entrenamiento, sin mirar fechas finales."""
    scored = []
    for threshold in np.arange(.25, .751, .025):
        metric = counts(calibration.positive,
                        calibration.probability.ge(threshold).to_numpy())
        if metric["good_detected"] >= .5 * int(calibration.positive.sum()):
            scored.append((metric["balanced_accuracy"],
                           metric["bad_rejected"], -abs(threshold - .5),
                           float(threshold), metric))
    if not scored:
        return .5, counts(calibration.positive,
                            calibration.probability.ge(.5).to_numpy())
    _, _, _, threshold, metric = max(scored)
    return threshold, metric


def evaluate(examples: pd.DataFrame, seconds: int = 30,
             eligible_only: bool = False) -> tuple[dict, pd.DataFrame]:
    frame = feature_table(examples, seconds)
    if eligible_only:
        frame = frame.loc[frame.distance_outside_target_pct.gt(0)].copy()
    dates = sorted(frame.session_date.unique())
    if len(dates) < 20:
        raise ValueError("Se necesitan al menos 20 fechas con etiquetas y cortes reales.")
    boundary = dates[int(len(dates) * .75)]
    train = frame.loc[frame.session_date < boundary].copy()
    test = frame.loc[frame.session_date >= boundary].copy()
    if train.positive.nunique() < 2 or test.positive.nunique() < 2:
        raise ValueError("Ambos períodos necesitan etiquetas buenas y malas.")
    sets = {"base": BASE_FEATURES, "width": WIDTH_FEATURES,
            "score_correction": SCORE_CORRECTION_FEATURES}
    if seconds > 10:
        sets["trajectory"] = TRAJECTORY_FEATURES
        sets["open_geometry"] = OPEN_GEOMETRY_FEATURES
    results = []
    for feature_name, features in sets.items():
        for model_name, model in candidates().items():
            calibration = inner_predictions(model, train, features)
            if calibration.empty or calibration.positive.nunique() < 2:
                continue
            threshold, inner_metric = select_threshold(calibration)
            fitted = clone(model)
            fitted.fit(train[list(features)], train.positive)
            probability = fitted.predict_proba(test[list(features)])[:, 1]
            results.append({"feature_set": feature_name, "model": model_name,
                            "threshold": round(threshold, 3),
                            "inner_cases": len(calibration),
                            "inner_accuracy": inner_metric["balanced_accuracy"],
                            "inner_counts": inner_metric,
                            "test_auc": round(float(roc_auc_score(test.positive,
                                                                  probability)), 3),
                            "test_counts": counts(test.positive, probability >= threshold),
                            "probability": probability})
    if not results:
        raise ValueError("No hay pliegues temporales útiles.")
    chosen = max(results, key=lambda row: (row["inner_accuracy"],
                                           row["inner_counts"]["bad_rejected"],
                                           row["feature_set"] == "base"))
    predictions = test[["session_date", "ticker", "label", "positive",
                        "target_band", "gap_abs_pct", "distance_outside_target_pct",
                        "target_angle_deg_5m_equiv", "provisional_width_ratio",
                        "existing_score_at_cutoff"]].copy()
    predictions["model_score"] = (chosen["probability"] * 100).round(1)
    predictions["selected"] = chosen["probability"] >= chosen["threshold"]
    report = {
        "cutoff_seconds": seconds,
        "eligible_only": eligible_only,
        "label_definition": "Excelente/Buena frente a Mala, juicios manuales posteriores",
        "train_dates": [train.session_date.min(), train.session_date.max()],
        "test_dates": [test.session_date.min(), test.session_date.max()],
        "train_cases": len(train), "test_cases": len(test),
        "test_labels": test.label.value_counts().to_dict(),
        "score_actual_ge70": counts(test.positive,
                                   test.existing_score_at_cutoff.ge(70).to_numpy()),
        "selected_by_inner_dates": {key: value for key, value in chosen.items()
                                    if key != "probability"},
        "all_experiments": [{key: value for key, value in row.items()
                             if key != "probability"} for row in results],
        "limitations": ["Las etiquetas no son una muestra aleatoria del universo.",
                        "Este período final ya se inspeccionó en estudios previos; "
                        "no es una prueba completamente nueva.",
                        "La hora SIP histórica no mide latencia del proveedor en vivo."],
    }
    return report, predictions


def rolling_evaluation(examples: pd.DataFrame, seconds: int = 20,
                       feature_set: tuple[str, ...] = TRAJECTORY_FEATURES,
                       model_name: str = "random_forest") -> dict:
    """Reentrena al avanzar las fechas y calibra el umbral sólo con el pasado."""
    frame = feature_table(examples, seconds)
    frame = frame.loc[frame.distance_outside_target_pct.gt(0)].copy()
    days = sorted(frame.session_date.unique())
    first = max(20, len(days) // 2)
    step = max(5, (len(days) - first) // 4)
    folds = []
    predictions = []
    for stop in range(first, len(days), step):
        earlier = frame.loc[frame.session_date.isin(days[:stop])]
        later = frame.loc[frame.session_date.isin(days[stop:stop + step])].copy()
        if earlier.positive.nunique() < 2 or later.positive.nunique() < 2:
            continue
        model = candidates()[model_name]
        calibration = inner_predictions(model, earlier, feature_set)
        if calibration.empty or calibration.positive.nunique() < 2:
            continue
        threshold, _ = select_threshold(calibration)
        fitted = clone(model).fit(earlier[list(feature_set)], earlier.positive)
        later["probability"] = fitted.predict_proba(later[list(feature_set)])[:, 1]
        later["selected"] = later.probability.ge(threshold)
        fold = {"train_end": days[stop - 1],
                "test_dates": [later.session_date.min(), later.session_date.max()],
                "threshold": round(threshold, 3), "cases": len(later),
                "model": counts(later.positive, later.selected),
                "original_ge70": counts(later.positive,
                                        later.existing_score_at_cutoff.ge(70))}
        folds.append(fold)
        predictions.append(later)
    if not predictions:
        raise ValueError("No se pudieron construir pliegues temporales.")
    tested = pd.concat(predictions, ignore_index=True)
    date_groups = [group.index.to_numpy() for _, group in
                   tested.groupby("session_date", sort=True)]
    random = np.random.default_rng(42)
    gains = []
    for _ in range(2000):
        chosen_days = random.integers(0, len(date_groups), len(date_groups))
        sampled = tested.iloc[np.concatenate([date_groups[index] for index in chosen_days])]
        if sampled.positive.nunique() < 2:
            continue
        model_accuracy = balanced_accuracy_score(sampled.positive, sampled.selected)
        old_accuracy = balanced_accuracy_score(
            sampled.positive, sampled.existing_score_at_cutoff.ge(70))
        gains.append(model_accuracy - old_accuracy)
    return {"seconds": seconds, "model": model_name,
            "feature_set": list(feature_set), "folds": folds,
            "cases": len(tested),
            "date_range": [tested.session_date.min(), tested.session_date.max()],
            "model_total": counts(tested.positive, tested.selected),
            "original_ge70_total": counts(tested.positive,
                                          tested.existing_score_at_cutoff.ge(70)),
            "paired_gain_balanced_accuracy": round(float(
                balanced_accuracy_score(tested.positive, tested.selected) -
                balanced_accuracy_score(tested.positive,
                                        tested.existing_score_at_cutoff.ge(70))), 3),
            "date_bootstrap_gain_ci95": [round(float(value), 3) for value in
                                          np.quantile(gains, [.025, .975])]}


def build_trial_history_20s(examples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Puntúa el universo histórico con el modelo experimental de 20 segundos.

    Es una vista retrospectiva para revisión manual. El modelo se ajusta con
    todos los juicios disponibles; sus scores históricos no son un backtest.
    """
    labeled = feature_table(examples, 20)
    labeled = labeled.loc[labeled.distance_outside_target_pct.gt(0)].copy()
    if len(labeled) < 30 or labeled.positive.nunique() < 2:
        raise ValueError("Faltan juicios con corte de 20 s fuera de Bollinger.")
    model = candidates()["random_forest"]
    calibration = inner_predictions(model, labeled, TRAJECTORY_FEATURES)
    if calibration.empty or calibration.positive.nunique() < 2:
        raise ValueError("Faltan fechas anteriores para calibrar el modelo de 20 s.")
    threshold, _ = select_threshold(calibration)
    model.fit(labeled[list(TRAJECTORY_FEATURES)], labeled.positive)

    setups, candles = export_history(100)
    days_with_trades = {path.stem.removeprefix("apertura_historica_30s_")
                       for path in OUTPUT_DIR.glob("apertura_historica_30s_*.csv")}
    days = sorted(set(setups.session_date) & days_with_trades)[-50:]
    base = setups.loc[setups.session_date.isin(days),
                      ["session_date", "ticker", "gap_direction"]].copy()
    observations = []
    for day in days:
        day_setups = base.loc[base.session_date.eq(day)]
        day_candles = candles.loc[candles.session_date.eq(day)]
        day_observed = early_observations(day)
        available = set(day_observed.loc[day_observed.window_seconds.eq(20), "ticker"])
        for ticker in day_setups.ticker:
            if ticker not in available:
                continue
            features = early_features(day, ticker, day_candles, day_observed)
            if not features.empty:
                observations.append(features.loc[features.window_seconds.isin((10, 20))])
    if not observations:
        raise ValueError("No hay operaciones históricas utilizables a 20 s.")
    universe = feature_table(pd.concat(observations, ignore_index=True), 20)
    universe["model_index_raw_20s"] = (
        model.predict_proba(universe[list(TRAJECTORY_FEATURES)])[:, 1] * 100).round(1)
    universe["existing_score_at_20s"] = universe.apply(score_at_cutoff, axis=1)
    eligible = universe.distance_outside_target_pct.gt(0)
    universe["model_score_20s"] = np.nan
    universe.loc[eligible, "model_score_20s"] = experimental_score(
        universe.loc[eligible, "model_index_raw_20s"].to_numpy() / 100,
        threshold).round(1)

    result = base.merge(universe, on=["session_date", "ticker"], how="left",
                        validate="one_to_one")
    reviews = load_reviews()
    result = result.merge(reviews[["session_date", "ticker", "label"]],
                          on=["session_date", "ticker"], how="left",
                          validate="one_to_one")
    result["manual_label"] = result.label.fillna("Pendiente")
    result["early_data_status"] = np.where(result.price.notna(), "Disponible", "Sin 20 s")
    expected = result.gap_direction.map({"UP": "UPPER", "DOWN": "LOWER"})
    direction_match = expected.eq(result.target_band)
    result["trial_candidate_20s"] = (
        result.price.notna() & result.distance_outside_target_pct.gt(0) &
        direction_match & result.model_score_20s.ge(70))
    result["trial_signal_status_20s"] = np.select(
        [result.price.isna(), ~direction_match,
         result.distance_outside_target_pct.le(0), result.model_score_20s.lt(70)],
        ["Sin operación antes de 20 s", "Cambió dirección del gap",
         "Dentro de Bollinger a 20 s", "Índice del modelo menor que 70"],
        default="Señal modelo 20 s")
    result["trial_failed_conditions_20s"] = np.where(
        result.trial_candidate_20s, "", result.trial_signal_status_20s)
    training_keys = set(zip(labeled.session_date, labeled.ticker))
    result["model_training_example"] = [
        (day, ticker) in training_keys for day, ticker in
        zip(result.session_date, result.ticker)]
    metadata = {
        "model": "random_forest", "feature_set": "trajectory", "cutoff_seconds": 20,
        "threshold_raw": round(float(threshold), 3),
        "training_cases": len(labeled), "training_dates": int(labeled.session_date.nunique()),
        "history_days": len(days), "history_rows": len(result),
        "with_20s": int(result.price.notna().sum()),
        "signals_ge70": int(result.trial_candidate_20s.sum()),
        "warning": ("Ajuste retrospectivo: el modelo utiliza todos los juicios disponibles, "
                    "incluso algunos posteriores a cada fecha y algunos de estos mismos "
                    "casos. Consultar evaluación temporal separada para desempeño."),
    }
    return result.sort_values(["session_date", "model_score_20s", "ticker"],
                              ascending=[False, False, True],
                              na_position="last").reset_index(drop=True), metadata


def main() -> None:
    _, examples = labeled_examples()
    if examples.empty:
        raise SystemExit("No hay ejemplos etiquetados con cortes reales.")
    stamp = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    summary = {}
    for seconds in (10, 20, 30):
        report, predictions = evaluate(examples, seconds, eligible_only=True)
        summary[str(seconds)] = report
        predictions.to_csv(OUTPUT_DIR / f"cutoff_model_predictions_{seconds}s_{stamp}.csv",
                           index=False)
        selected = report["selected_by_inner_dates"]
        print(f"{seconds}s: {selected['feature_set']} / {selected['model']} "
              f"({selected['test_counts']})", flush=True)
    summary["rolling_20s"] = rolling_evaluation(examples)
    path = OUTPUT_DIR / f"cutoff_model_study_eligible_{stamp}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    history, metadata = build_trial_history_20s(examples)
    history.to_csv(OUTPUT_DIR / f"cutoff_model_history_20s_{stamp}.csv", index=False)
    (OUTPUT_DIR / f"cutoff_model_history_20s_{stamp}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: {path}")
    print(f"Historial de prueba: {metadata['history_days']} sesiones, "
          f"{metadata['signals_ge70']} señales experimentales")


if __name__ == "__main__":
    main()
