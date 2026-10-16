"""Descripciones posteriores de la acción, sin precios de contratos ni modelo."""

import numpy as np
import pandas as pd


OUTCOME_COLUMNS = [
    "session_date", "source", "ticker", "gap_direction", "candidate_at_open",
    "confirmed", "touched_target_band", "touch_time", "touch_status",
    "reversion_to_touch_band_pct", "band_at_or_beyond_open_at_touch",
    "opening_price", "last_close", "move_to_last_close_pct",
    "best_observed_excursion_pct",
]


def build_outcomes(setups: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Evalúa ex post movimientos del subyacente; no representa una operación.

    Si la señal se confirmó a las 09:31 y sólo hay velas de 15 minutos, se
    excluye la vela 09:30–09:45 del máximo favorable: contiene minutos previos
    a la confirmación. El toque registrado por el escáner puede tener más
    precisión cuando existen velas de 1 minuto.
    """
    if setups.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    results = []
    grouped = ({key: group.loc[group.session == "ACTUAL"].sort_values("bar_index")
                for key, group in candles.groupby(["session_date", "source", "ticker"])}
               if not candles.empty else {})
    for setup in setups.itertuples():
        if pd.isna(setup.opening_price):
            continue
        rows = grouped.get((setup.session_date, setup.source, setup.ticker),
                           pd.DataFrame(columns=["bar_index", "high", "low", "close"]))
        sign = 1 if setup.gap_direction == "UP" else -1
        opening = float(setup.opening_price)
        last = float(rows.iloc[-1].close) if not rows.empty else np.nan
        movement = sign * (opening - last) / opening * 100 if np.isfinite(last) else np.nan
        eligible = rows.iloc[1:] if int(setup.confirmed) == 1 else rows
        if eligible.empty:
            best = np.nan
        elif sign == 1:
            best = (opening - float(eligible.low.min())) / opening * 100
        else:
            best = (float(eligible.high.max()) - opening) / opening * 100
        results.append({
            "session_date": setup.session_date, "source": setup.source,
            "ticker": setup.ticker, "gap_direction": setup.gap_direction,
            "candidate_at_open": bool(setup.candidate_at_open),
            "confirmed": int(setup.confirmed),
            "touched_target_band": bool(setup.touched_target_band),
            "touch_time": setup.touch_time, "touch_status": setup.touch_status,
            "reversion_to_touch_band_pct": setup.reversion_to_touch_band_pct,
            "band_at_or_beyond_open_at_touch": setup.band_at_or_beyond_open_at_touch,
            "opening_price": opening, "last_close": last,
            "move_to_last_close_pct": movement,
            "best_observed_excursion_pct": best,
        })
    return pd.DataFrame.from_records(results, columns=OUTCOME_COLUMNS)


def summary(outcomes: pd.DataFrame) -> dict:
    if outcomes.empty:
        return {"cases": 0, "touched": 0, "observed_positive_close": 0}
    return {
        "cases": len(outcomes),
        "touched": int(outcomes.touched_target_band.sum()),
        "observed_positive_close": int((outcomes.move_to_last_close_pct > 0).sum()),
    }
