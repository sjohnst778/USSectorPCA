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

    prices = prices.dropna(how="all")

    # Drop tickers that returned entirely empty data (e.g. rate-limited by Yahoo)
    all_null = prices.columns[prices.isnull().all()]
    if len(all_null):
        prices = prices.drop(columns=all_null)

    return prices
