import yfinance as yf
import pandas as pd


def fetch_etf_holdings(ticker: str) -> pd.DataFrame:
    """Return top holdings for an ETF as a DataFrame.

    Columns returned: symbol, name, weight (0–100 scale).
    yfinance returns up to 10 holdings with the index as the ticker symbol.
    Returns an empty DataFrame if holdings are unavailable.
    """
    try:
        raw = yf.Ticker(ticker).funds_data.top_holdings
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["symbol", "name", "weight"])

        data = raw.reset_index()  # moves Symbol index to column
        data.columns = ["symbol", "name", "weight"]
        data["symbol"] = data["symbol"].str.strip().str.upper()

        # weight stored as fraction (0–1) → convert to percent
        if data["weight"].max() <= 1.0:
            data["weight"] = data["weight"] * 100

        return data.reset_index(drop=True)

    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "weight"])
