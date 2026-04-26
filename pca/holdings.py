import time

import yfinance as yf
import pandas as pd

_EMPTY = pd.DataFrame(columns=["symbol", "name", "weight"])


def fetch_etf_holdings(ticker: str, retries: int = 2, retry_delay: float = 2.0) -> pd.DataFrame:
    """Return top holdings for an ETF as a DataFrame.

    Columns returned: symbol, name, weight (0–100 scale).
    yfinance returns up to 10 holdings with the index as the ticker symbol.
    Retries on failure to handle transient Yahoo Finance rate-limiting.
    Returns an empty DataFrame if holdings are unavailable after retries.
    """
    for attempt in range(retries + 1):
        try:
            raw = yf.Ticker(ticker).funds_data.top_holdings
            if raw is None or raw.empty:
                if attempt < retries:
                    time.sleep(retry_delay)
                    continue
                return _EMPTY

            data = raw.reset_index()  # moves Symbol index to column
            data.columns = ["symbol", "name", "weight"]
            data["symbol"] = data["symbol"].str.strip().str.upper()

            # weight stored as fraction (0–1) → convert to percent
            if data["weight"].max() <= 1.0:
                data["weight"] = data["weight"] * 100

            return data.reset_index(drop=True)

        except Exception:
            if attempt < retries:
                time.sleep(retry_delay)

    return _EMPTY
