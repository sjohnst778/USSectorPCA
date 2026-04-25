import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute return series from adjusted close prices.

    method: 'log' for log returns, 'simple' for arithmetic returns.
    Drops any ticker with more than 5% missing observations after the intersection.
    """
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()

    rets = rets.iloc[1:]
    threshold = 0.05 * len(rets)
    rets = rets.dropna(axis=1, thresh=int(len(rets) - threshold))
    return rets.dropna()


def compute_relative_returns(
    sector_returns: pd.DataFrame, benchmark_returns: pd.Series
) -> pd.DataFrame:
    """Subtract benchmark (e.g. SPY) return from each sector return on each date.

    Both inputs must share a DatetimeIndex. The intersection of dates is used.
    """
    common = sector_returns.index.intersection(benchmark_returns.index)
    rel = sector_returns.loc[common].subtract(benchmark_returns.loc[common], axis=0)
    return rel.dropna()
