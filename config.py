"""Configuración local del MVP Efecto Imán."""

from pathlib import Path
from hashlib import sha256


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "efecto_iman.db"
HISTORY_SESSION_LIMIT = 100

# Símbolos del listado original y las ampliaciones solicitadas; no se usan sus precios.
TICKERS = (
    "AMZN", "AAPL", "GOOG", "META", "MSFT", "NFLX", "TSLA", "PLTR", "ORCL",
    "AMD", "MU", "NVDA", "QCOM", "AVGO", "SOXL", "DASH", "LYFT", "UBER",
    "HD", "LOW", "WMT", "DIA", "QQQ", "SPY", "SPX", "IWM", "TNA", "AXP",
    "C", "MA", "PYPL", "V", "BABA", "LI", "NIO", "XPEV", "GLD", "SLV",
    "USO", "COIN", "HOOD", "CVS", "MRNA", "PFE", "BA", "URA", "CCL",
    "RCL", "AAL", "DAL", "TZA",
    "JNJ", "DIS", "MRVL", "IBM", "JPM", "XOM", "CAT", "SBUX", "VZ",
    "T", "MCD", "NKE", "KO", "SCHW", "COST", "DELL", "WFC", "ANET",
    "PG", "CVX", "MRK",
)

# El proveedor llama ^GSPC al índice que aparece como SPX en el listado.
PROVIDER_SYMBOLS = {"SPX": "^GSPC"}
SCAN_VERSION = "target-angle-v3"
UNIVERSE_ID = sha256((SCAN_VERSION + "|" + "|".join(TICKERS)).encode("ascii")).hexdigest()[:12]
