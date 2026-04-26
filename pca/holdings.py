import time

import yfinance as yf
import pandas as pd

_EMPTY = pd.DataFrame(columns=["symbol", "name", "weight"])


def fetch_etf_holdings(ticker: str, retries: int = 4, retry_delay: float = 3.0) -> pd.DataFrame:
    """Return top holdings for an ETF as a DataFrame.

    Columns returned: symbol, name, weight (0–100 scale).
    yfinance returns up to 10 holdings with the index as the ticker symbol.
    Retries with exponential backoff to handle Yahoo Finance rate-limiting,
    which is more aggressive for European (.DE) ETFs on cloud IPs.
    Returns an empty DataFrame if holdings are unavailable after retries.
    """
    for attempt in range(retries + 1):
        try:
            raw = yf.Ticker(ticker).funds_data.top_holdings
            if raw is None or raw.empty:
                if attempt < retries:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return _EMPTY

            data = raw.reset_index()  # moves Symbol index to column
            data.columns = ["symbol", "name", "weight"]
            # Preserve exchange suffix case (.AS, .DE, .L etc.) — only strip whitespace
            data["symbol"] = data["symbol"].str.strip()

            # weight stored as fraction (0–1) → convert to percent
            if data["weight"].max() <= 1.0:
                data["weight"] = data["weight"] * 100

            return data.reset_index(drop=True)

        except Exception:
            if attempt < retries:
                time.sleep(retry_delay * (attempt + 1))

    return _EMPTY
