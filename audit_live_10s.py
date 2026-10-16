"""Freeze and explain a real 09:30:10 capture without downloading later prices.

The opening CSV is the evidence of timely receipt. Only PREVIA 15-minute
candles from the local session are used to reconstruct the opening score.
Running the command again leaves the frozen opening files unchanged and updates
only the separate comparison with the latest saved diagnostic.
"""

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import DATA_DIR, TICKERS
from daily_store import load_daily_session
from opening_scan import opening_setups


NEW_YORK = ZoneInfo("America/New_York")
SNAPSHOTS = DATA_DIR / "snapshots"
OPEN_COMPONENTS = (
    "score_gap_component", "score_distance_component",
    "score_compression_component", "score_prior_width_growth_component",
)
FROZEN_COLUMNS = (
    "date", "ticker", "observed_within_10s", "price_at_cutoff",
    "quote_event_time", "quote_received_at", "score_open", "candidate_at_open",
    "gap_direction", "gap_pct", "opening_price", "prior_close", "target_band",
    "prior_bb_upper", "prior_bb_lower", "cutoff_bb_upper", "cutoff_bb_lower",
    "cutoff_target_band", "cutoff_target_band_reaction_pct",
    "distance_to_band_pct", "prior_distance_to_band_pct",
    "target_angle_10s_deg", "angle_10s_event_span_seconds",
    "prior_bb_width_pct", "prior_bb_width_change",
    "prior_target_band_slope_pct", *OPEN_COMPONENTS,
    "score_target_angle_component", "failed_conditions", "pending_conditions",
    "data_status",
)
LATER_COLUMNS = (
    "score_open", "score_1m", "score_5m", "score_15m", "score",
    "signal_stage", "candidate", "gap_pct", "opening_price",
    "target_angle_1m_deg", "target_angle_5m_deg",
    "target_angle_0935_0940", "target_angle_0940_0945",
    "post_open_bb_expansion_pct", "distance_to_band_after_15m",
    "touched_target_band", "touch_status", "failed_conditions",
)


def timely_quotes(day: str) -> pd.DataFrame:
    path = SNAPSHOTS / f"apertura_10s_{day}.csv"
    quotes = pd.read_csv(path)
    required = {"ticker", "price_at_cutoff", "quote_event_time", "quote_received_at"}
    if not required.issubset(quotes.columns):
        raise ValueError(f"Captura incompleta: {path}")
    if quotes.ticker.duplicated().any():
        raise ValueError("La captura tiene símbolos duplicados")
    opened = datetime.combine(date.fromisoformat(day), time(9, 30), NEW_YORK)
    received = pd.to_datetime(quotes.quote_received_at, utc=True)
    event = pd.to_datetime(quotes.quote_event_time, utc=True)
    if not ((received >= opened) & (received <= opened + timedelta(seconds=10))).all():
        raise ValueError("La captura incluye una recepción fuera de los diez segundos")
    if not ((event >= opened) & (event <= received + timedelta(seconds=3))).all():
        raise ValueError("La captura incluye una marca del proveedor fuera del corte")
    return quotes


def freeze_opening(day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    quotes = timely_quotes(day)
    saved, candles, _ = load_daily_session(day)
    prior = candles.loc[candles.session == "PREVIA"].copy()
    if prior.empty or prior.ticker.nunique() != len(TICKERS):
        raise RuntimeError("Faltan velas previas locales para reconstruir el universo")
    prices = dict(zip(quotes.ticker, quotes.price_at_cutoff))
    events: dict[str, list[tuple[datetime, float]]] = {}
    stream_path = SNAPSHOTS / f"apertura_flujo_30s_{day}.csv"
    if stream_path.exists():
        stream = pd.read_csv(stream_path)
        opened = datetime.combine(date.fromisoformat(day), time(9, 30), NEW_YORK)
        cutoff = opened + timedelta(seconds=10)
        for row in stream.itertuples():
            event_time = datetime.fromisoformat(row.quote_event_time)
            received_at = datetime.fromisoformat(row.quote_received_at)
            if (opened <= event_time <= cutoff and received_at <= cutoff
                    and row.ticker in prices):
                events.setdefault(row.ticker, []).append((event_time, float(row.price)))
    reconstructed = opening_setups(prior, opening_prices=prices, cutoff_seconds=10,
                                   opening_events=events)

    # El registro escrito en el corte permite comprobar la reconstrucción de
    # todos los candidatos que fueron visibles entonces.
    audit_path = DATA_DIR / "signal_availability.csv"
    if audit_path.exists():
        recorded = pd.read_csv(audit_path)
        recorded = recorded.loc[(recorded.date == day) &
                                (recorded.source == "corte_10s")]
        if not recorded.empty:
            candidates = reconstructed.loc[reconstructed.candidate_at_open]
            if set(candidates.ticker) != set(recorded.ticker):
                raise ValueError("Los candidatos reconstruidos difieren del registro del corte")
            scores = candidates.set_index("ticker").score_open
            original = recorded.set_index("ticker").score_open
            if ((scores - original).abs() > 0.05).any():
                raise ValueError("El score reconstruido difiere del registro del corte")

    result = reconstructed.merge(quotes, on="ticker", how="left", validate="one_to_one")
    result.insert(0, "date", day)
    result["observed_within_10s"] = result.price_at_cutoff.notna()
    result = result.reindex(columns=FROZEN_COLUMNS)
    result = result.sort_values(["candidate_at_open", "score_open", "ticker"],
                                ascending=[False, False, True], na_position="last")
    return result, saved


def save_audit(day: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    full_path = SNAPSHOTS / f"diagnostico_apertura_10s_{day}.csv"
    candidates_path = SNAPSHOTS / f"candidatos_apertura_10s_{day}.csv"
    comparison_path = SNAPSHOTS / f"comparacion_apertura_10s_{day}.csv"
    if full_path.exists() and candidates_path.exists():
        frozen = pd.read_csv(full_path)
        current, _, updated = load_daily_session(day)
    else:
        frozen, current = freeze_opening(day)
        _, _, updated = load_daily_session(day)
        frozen.to_csv(full_path, index=False, float_format="%.8f")
        frozen.loc[frozen.candidate_at_open].to_csv(
            candidates_path, index=False, float_format="%.8f")
    later = current.reindex(columns=["ticker", *LATER_COLUMNS]).rename(
        columns={column: f"later_{column}" for column in LATER_COLUMNS})
    comparison = frozen.merge(later, on="ticker", how="left", validate="one_to_one")
    opened = datetime.combine(date.fromisoformat(day), time(9, 30), NEW_YORK)
    event = pd.to_datetime(comparison.quote_event_time, utc=True)
    received = pd.to_datetime(comparison.quote_received_at, utc=True)
    comparison["event_second_after_open"] = (event - opened).dt.total_seconds()
    comparison["received_second_after_open"] = (received - opened).dt.total_seconds()
    comparison["delivery_lag_seconds"] = (received - event).dt.total_seconds()
    comparison["later_session_updated_at"] = updated
    comparison.to_csv(comparison_path, index=False, float_format="%.8f")
    return frozen, comparison, full_path, candidates_path, comparison_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the live ten-second opening capture")
    parser.add_argument("day", nargs="?", default=datetime.now(NEW_YORK).date().isoformat())
    day = parser.parse_args().day
    frozen, comparison, full, candidates, later = save_audit(day)
    observed = frozen.loc[frozen.observed_within_10s]
    chosen = observed.loc[observed.candidate_at_open]
    print(f"{day}: {len(observed)}/{len(frozen)} cotizaciones a tiempo; "
          f"{len(chosen)} candidatos preliminares; {chosen.score_open.ge(70).sum()} con score >=70")
    print(chosen[["ticker", "score_open", "gap_pct", "distance_to_band_pct",
                  *OPEN_COMPONENTS]].to_string(index=False))
    print(f"Guardado: {full}\nGuardado: {candidates}\nComparación actualizada: {later}")
    complete = comparison.later_score_15m.notna().sum()
    print(f"Scores de 15m disponibles: {complete}/{len(comparison)}")


if __name__ == "__main__":
    main()
