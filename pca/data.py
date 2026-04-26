import time

import yfinance as yf
import pandas as pd


def _download_single(ticker: str, start: str, end: str) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"].squeeze()
    return raw["Close"].squeeze()


def fetch_adjusted_close(
    tickers: list[str], start: str, end: str, retries: int = 2, retry_delay: float = 2.0
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Retries any tickers that return entirely empty data (common when Yahoo
    Finance rate-limits cloud datacenter IPs during a batch download).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")

    failed = [t for t in tickers if t not in prices.columns or prices[t].isnull().all()]
    for attempt in range(retries):
        if not failed:
            break
        time.sleep(retry_delay)
        still_failed = []
        for ticker in failed:
            try:
                s = _download_single(ticker, start, end)
                if s is not None and not s.isnull().all():
                    prices[ticker] = s
                else:
                    still_failed.append(ticker)
            except Exception:
                still_failed.append(ticker)
        failed = still_failed

    # Remove any columns that are still all-null after retries
    all_null = [t for t in prices.columns if prices[t].isnull().all()]
    if all_null:
        prices = prices.drop(columns=all_null)

    return prices
