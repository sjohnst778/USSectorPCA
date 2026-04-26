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


def compute_period_stats(
    returns: pd.DataFrame,
    method: str = "log",
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute total return, annualised return, annualised volatility, beta, and risk-adjusted return.

    Beta is the OLS slope of each asset's returns on the benchmark series.
    Risk-adjusted return = annualised return / annualised volatility (Sharpe without risk-free rate).
    All values returned as decimals.
    """
    n = len(returns)

    if method == "log":
        total_return = np.exp(returns.sum()) - 1
    else:
        total_return = (1 + returns).prod() - 1

    ann_return = (1 + total_return) ** (252 / n) - 1
    ann_vol = returns.std() * np.sqrt(252)
    risk_adjusted = ann_return / ann_vol.replace(0, np.nan)

    if benchmark is not None:
        common = returns.index.intersection(benchmark.index)
        bm = benchmark.loc[common]
        bm_var = bm.var()
        beta = pd.Series(
            {col: returns[col].loc[common].cov(bm) / bm_var for col in returns.columns},
            name="beta",
        )
    else:
        beta = pd.Series(np.nan, index=returns.columns, name="beta")

    return pd.DataFrame({
        "beta": beta,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "risk_adjusted": risk_adjusted,
    })


def compute_relative_returns(
    sector_returns: pd.DataFrame, benchmark_returns: pd.Series
) -> pd.DataFrame:
    """Subtract benchmark (e.g. SPY) return from each sector return on each date.

    Both inputs must share a DatetimeIndex. The intersection of dates is used.
    """
    common = sector_returns.index.intersection(benchmark_returns.index)
    rel = sector_returns.loc[common].subtract(benchmark_returns.loc[common], axis=0)
    return rel.dropna()
