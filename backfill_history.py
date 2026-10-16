"""Carga en SQLite las sesiones históricas disponibles del universo configurado."""

from config import HISTORY_SESSION_LIMIT, TICKERS
from daily_store import initialize_daily_store, save_daily_session
from real_data import available_days, download_bars, scan_real_day


def main() -> None:
    initialize_daily_store()
    bars, missing = download_bars(period="59d", tickers=TICKERS)
    print(f"Símbolos con velas de 15 min: {len(bars)}/{len(TICKERS)}", flush=True)
    if missing:
        print("Sin velas: " + ", ".join(missing), flush=True)
    try:
        minute_bars, missing_minutes = download_bars(period="7d", tickers=TICKERS, interval="1m")
    except Exception:
        minute_bars, missing_minutes = {}, TICKERS
    print(f"Símbolos con velas de 1 min: {len(minute_bars)}/{len(TICKERS)}", flush=True)
    if missing_minutes:
        print("Sin velas de 1 min: " + ", ".join(missing_minutes), flush=True)
    five_minute_bars, missing_five = download_bars(period="59d", tickers=TICKERS,
                                                    interval="5m")
    print(f"Símbolos con velas de 5 min: {len(five_minute_bars)}/{len(TICKERS)}", flush=True)
    if missing_five:
        print("Sin velas de 5 min: " + ", ".join(missing_five), flush=True)
    days = available_days(bars)[-HISTORY_SESSION_LIMIT:]
    saved = 0
    candidates_total = 0
    for day in days:
        setups, candles = scan_real_day(bars, day, minute_bars, five_minute_bars)
        if candles.empty:
            continue
        save_daily_session(day, setups, candles, replace_existing=True)
        saved += 1
        candidates = int(setups.candidate.sum()) if not setups.empty else 0
        candidates_total += candidates
        print(f"{day}: {len(setups)} símbolos diagnosticados; {candidates} candidatos", flush=True)
    print(f"Sesiones guardadas: {saved}/{len(days)}; candidatos: {candidates_total}", flush=True)


if __name__ == "__main__":
    main()
