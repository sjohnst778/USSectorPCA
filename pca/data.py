import yfinance as yf
import pandas as pd


def fetch_adjusted_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices.dropna(how="all")
