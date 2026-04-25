# PCA Equity Analysis — Project Reference

## Overview
A Python application for analysing equity returns using Principal Component Analysis (PCA). Data is sourced from Yahoo Finance. Results are presented via an interactive Streamlit dashboard.

**Goal:** Allow users to select a basket of equities, retrieve historical price data, compute PCA on the return series, and interpret the resulting components — identifying latent factors driving cross-sectional return variation.

---

## Architecture

```
PCA/
├── CLAUDE.md
├── requirements.txt
├── app.py                  # Streamlit entry point
├── pca/
│   ├── __init__.py
│   ├── data.py             # Yahoo Finance data retrieval
│   ├── returns.py          # Return computation (log/simple)
│   ├── analysis.py         # PCA logic (sklearn / numpy)
│   └── plots.py            # Chart helpers (plotly)
└── .venv/
```

### Key Components
| Module | Responsibility |
|---|---|
| `pca/data.py` | Fetch OHLCV data via `yfinance`; cache to avoid redundant downloads |
| `pca/returns.py` | Compute simple or log returns from adjusted close prices |
| `pca/analysis.py` | Standardise returns, fit PCA, expose loadings / explained variance |
| `pca/plots.py` | Plotly charts: scree plot, biplot, factor loadings heatmap, cumulative explained variance |
| `app.py` | Streamlit UI: ticker input, date range, PCA controls, chart display |

---

## Stack
| Tool | Purpose |
|---|---|
| Python 3.13 | Runtime |
| `yfinance` | Yahoo Finance data retrieval |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | PCA implementation |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Usage
1. Enter a comma-separated list of ticker symbols (e.g. `AAPL, MSFT, GOOGL`)
2. Select a date range
3. Choose return type (log or simple) and lookback window
4. Select number of principal components to retain
5. Inspect scree plot, loadings heatmap, and biplot

---

## Dual PCA — Absolute vs Relative
The app runs two parallel PCAs side by side:

| Mode | Input | What it reveals |
|---|---|---|
| **Absolute** | Raw sector ETF returns | PC1 = broad market factor; subsequent PCs = residual variation |
| **Relative** | Sector return − SPY return (daily) | Market factor removed; PCs reveal rotation: cyclical vs defensive, momentum, etc. |

**Benchmark:** SPY (S&P 500 ETF) — consistent with using ETFs throughout.

### S&P 500 Sector Preset
| Ticker | Sector | History from |
|---|---|---|
| XLB | Materials | 1998 |
| XLC | Communication Services | **2018-06-19** |
| XLE | Energy | 1998 |
| XLF | Financials | 1998 |
| XLI | Industrials | 1998 |
| XLK | Information Technology | 1998 |
| XLP | Consumer Staples | 1998 |
| XLRE | Real Estate | **2015-10-07** |
| XLU | Utilities | 1998 |
| XLV | Health Care | 1998 |
| XLY | Consumer Discretionary | 1998 |

XLC and XLRE have limited history; the app warns the user if the selected date range predates them.

---

## Design Decisions & Notes
- **Adjusted close** prices used throughout to account for splits and dividends.
- **Standardisation** (zero mean, unit variance) applied before PCA so that high-volatility stocks don't dominate components.
- Returns are computed on the **overlapping intersection** of trading days across all selected tickers to avoid NaN contamination.
- **Log returns** default — additive, approximately normal, better suited for PCA.
- **Relative returns** = sector daily return − SPY daily return (not excess over risk-free rate). This is a pure cross-sectional view, not an alpha measure.
- **Standardise** (default on): scales each return series to unit variance before PCA. When on, covariance of scaled data = correlation matrix, making the matrix-type toggle moot.
- **PCA matrix type** (Correlation / Covariance): eigendecompose the correlation or covariance matrix. Only has a distinct effect when standardise is off; otherwise both resolve to the correlation matrix. The effective matrix used is shown as a metric in the UI.
- **Ledoit-Wolf shrinkage** (optional, default on): shrinks the sample matrix towards a scaled identity before eigendecomposition. The shrinkage coefficient α is shown as a UI metric. With 11 sectors and ~1000 observations α ≈ 0.01 (well-conditioned), but shrinkage is best practice and becomes material with shorter windows or larger baskets.

---

## Changelog
| Date | Change |
|---|---|
| 2026-04-25 | Project initialised — CLAUDE.md created, requirements defined, package scaffold created |
| 2026-04-25 | Added dual PCA (absolute vs relative), SPY benchmark, sector ETF preset, `compute_relative_returns` |
| 2026-04-25 | Added Ledoit-Wolf shrinkage option; PCA now runs on correlation matrix explicitly; shrinkage α surfaced in UI |
| 2026-04-25 | Added standardise toggle and correlation/covariance matrix selector; all paths unified through explicit eigendecomposition |
| 2026-04-25 | Added constituent drill-down: fetch ETF top holdings via yfinance, run constituent PCA relative to ETF, display PC1/PC2 loadings table sorted by \|PC1\| |
