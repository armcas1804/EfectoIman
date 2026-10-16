"""Compara una sesión exacta de 09:30:10 con score_15m y modelo previo."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from evaluate_historical_10s import SNAPSHOTS, load_joined_examples
from evaluate_rescue_10s import candidate_models, cross_validate, rescue_examples


def review_day(day: str) -> tuple[dict, pd.DataFrame]:
    path = SNAPSHOTS / f"apertura_historica_10s_{day}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta {path.name}. Descárgalo con backfill_10s_alpaca.py --date {day}")
    frame = load_joined_examples()
    today = frame.loc[frame.date == day].copy()
    if today.empty:
        raise ValueError("El archivo existe, pero no coincide con diagnósticos de ese día")
    previous = rescue_examples(frame.loc[frame.date < day])
    low = rescue_examples(today)
    if previous.date.nunique() < 20:
        raise ValueError("Faltan sesiones anteriores para entrenar el modelo")
    candidates = candidate_models()
    validation = {name: cross_validate(previous, model, features)
                  for name, (model, features) in candidates.items()}
    choice = max(candidates, key=lambda name: (
        validation[name]["average_precision"],
        validation[name]["top_3_per_day"]["aciertos"]))
    model, features = candidates[choice]
    model.fit(previous.loc[:, features], previous.label_15m_ge_70)
    today["rescue_probability"] = float("nan")
    if not low.empty:
        today.loc[low.index, "rescue_probability"] = model.predict_proba(
            low.loc[:, features])[:, 1]
    today["early_ge_70"] = today.score_10s_geometry >= 70
    today["later_ge_70"] = today.score_15m >= 70
    today["maintained_70"] = today.early_ge_70 & today.later_ge_70
    today["rescued_70"] = ~today.early_ge_70 & today.later_ge_70
    outside = today.distance_to_band_pct_10s > 0
    summary = {
        "date": day, "symbols_with_10s_and_diagnostics": len(today),
        "outside_band_at_10s": int(outside.sum()),
        "early_ge_70": int(today.early_ge_70.sum()),
        "later_ge_70": int(today.later_ge_70.sum()),
        "maintained_70": int(today.maintained_70.sum()),
        "rescued_70": int(today.rescued_70.sum()),
        "selected_model": choice,
        "model_training_last_date": previous.date.max(),
        "low_score_outside_count": len(low),
        "model_top_3_low_score": today.loc[low.index].sort_values(
            "rescue_probability", ascending=False).head(3)[
                ["ticker", "score_10s_geometry", "score_15m",
                 "rescue_probability"]].to_dict("records"),
        "note": ("La probabilidad del modelo sólo se aplica a score de 10s <70 "
                 "y apertura fuera de Bollinger. score_15m es el resultado "
                 "observado después y nunca se usa para entrenar en este día."),
    }
    return summary, today.sort_values("score_10s_geometry", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="fecha de sesión YYYY-MM-DD")
    args = parser.parse_args()
    try:
        summary, rows = review_day(args.date)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    (SNAPSHOTS / f"review_10s_{args.date}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rows.to_csv(SNAPSHOTS / f"review_10s_{args.date}.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nScores tempranos >=70:")
    columns = ["ticker", "score_10s_geometry", "score_15m", "maintained_70"]
    print(rows.loc[rows.early_ge_70, columns].to_string(index=False))


if __name__ == "__main__":
    main()
