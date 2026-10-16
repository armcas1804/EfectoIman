"""Indicadores descriptivos para velas bursátiles de 15 minutos."""

import numpy as np
import pandas as pd


BOLLINGER_WINDOW = 20
BOLLINGER_DEVIATIONS = 2


def percentage_distance(value: float, reference: float) -> float:
    """Distancia porcentual absoluta desde value hasta reference."""
    if value == 0:
        return 0.0
    return float(np.abs(reference - value) / np.abs(value) * 100)


def bollinger_bands(candles: pd.DataFrame) -> pd.DataFrame:
    """Agrega media y bandas de 20 cierres con dos desviaciones estándar."""
    result = candles.copy().sort_values("bar_index").reset_index(drop=True)
    close = result["close"].astype(float)
    middle = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
    deviation = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std(ddof=0)
    result["bb_middle"] = middle
    result["bb_upper"] = middle + BOLLINGER_DEVIATIONS * deviation
    result["bb_lower"] = middle - BOLLINGER_DEVIATIONS * deviation
    return result
