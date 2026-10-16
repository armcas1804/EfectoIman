"""Escanea aperturas con ángulo superior y exporta diagnósticos sin tocar SQLite."""

import argparse
from datetime import datetime

import pandas as pd

from config import DATA_DIR, TICKERS
from opening_scan import MINUTE_BAND_COLUMNS, UPPER_ANGLE_COLUMNS
from real_data import NEW_YORK, available_days, download_bars, scan_upper_angle_day


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=50,
                        help="Número máximo de sesiones disponibles (por defecto: 50)")
    args = parser.parse_args()
    bars, missing_15m = download_bars(period="59d", tickers=TICKERS)
    try:
        minutes, missing_1m = download_bars(period="7d", interval="1m", tickers=TICKERS)
    except Exception as exc:
        minutes, missing_1m = {}, TICKERS
        print(f"No se obtuvieron velas de 1 minuto: {exc}", flush=True)
    diagnoses, sequences = [], []
    today = datetime.now(NEW_YORK).date().isoformat()
    historical_days = [day for day in available_days(bars) if day < today]
    for day in historical_days[-args.days:]:
        setups, minute_series = scan_upper_angle_day(bars, day, minutes)
        diagnoses.append(setups)
        sequences.append(minute_series)
    all_setups = (pd.concat(diagnoses, ignore_index=True) if diagnoses else
                  pd.DataFrame(columns=UPPER_ANGLE_COLUMNS))
    all_minutes = (pd.concat(sequences, ignore_index=True) if sequences else
                   pd.DataFrame(columns=MINUTE_BAND_COLUMNS))
    if not all_setups.empty:
        all_setups = all_setups.sort_values(
            ["candidate", "distance_above_upper_pct", "date", "ticker"],
            ascending=[False, False, False, True]).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    diagnoses_path = DATA_DIR / "upper_angle_diagnostics.csv"
    sequence_path = DATA_DIR / "upper_band_minutes.csv"
    all_setups.to_csv(diagnoses_path, index=False)
    all_minutes.to_csv(sequence_path, index=False)
    candidates = all_setups.loc[all_setups.candidate == True]
    measured = int(all_setups.upper_angle_0930_0935.notna().sum())
    print(f"Sesiones: {all_setups.date.nunique()}; aperturas: {len(all_setups)}; "
          f"ángulo inicial medido: {measured}; candidatos: {len(candidates)}", flush=True)
    print(f"Sin datos de 15m: {len(missing_15m)}; sin datos de 1m: {len(missing_1m)}",
          flush=True)
    if not candidates.empty:
        columns = ["date", "ticker", "gap_pct", "distance_above_upper_pct",
                   "upper_angle_0930_0935", "upper_angle_0935_0940",
                   "upper_angle_0940_0945"]
        print(candidates[columns].head(12).to_string(index=False), flush=True)
    print(f"CSV: {diagnoses_path} y {sequence_path}", flush=True)


if __name__ == "__main__":
    main()
