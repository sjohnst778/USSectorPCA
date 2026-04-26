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
├── runtime.txt             # Python 3.12 — for Streamlit Community Cloud
├── .gitignore
├── app.py                  # Streamlit entry point
└── pca/
    ├── __init__.py
    ├── data.py             # Yahoo Finance price retrieval
    ├── returns.py          # Log/simple return computation; relative returns vs benchmark
    ├── analysis.py         # PCA via explicit eigendecomposition (LW shrinkage optional)
    ├── holdings.py         # ETF constituent lookup via yfinance funds_data
    └── plots.py            # Plotly charts: scree, loadings heatmap, biplot, correlation heatmap, PC score projections
```

### Key Components
| Module | Responsibility |
|---|---|
| `pca/data.py` | Fetch adjusted close prices via `yfinance` for a list of tickers |
| `pca/returns.py` | Compute log or simple returns; `compute_relative_returns` subtracts benchmark daily return |
| `pca/analysis.py` | Demean, optionally standardise, compute correlation/covariance matrix (with optional LW shrinkage), eigendecompose; returns `PCAResult` dataclass |
| `pca/holdings.py` | Fetch ETF top-10 holdings (symbol, name, weight) from `yfinance.Ticker.funds_data` |
| `pca/plots.py` | Scree plot, loadings heatmap, biplot, correlation heatmap, cumulative PC score projections |
| `app.py` | Streamlit UI — all state persisted in `st.session_state` to support independent button interactions |

---

## Stack
| Tool | Purpose |
|---|---|
| Python 3.12 | Runtime (3.13 locally; 3.12 specified for Streamlit Cloud via `runtime.txt`) |
| `yfinance` | Yahoo Finance data retrieval |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Ledoit-Wolf covariance estimator |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard |
| `matplotlib` | Required by pandas `.background_gradient()` for styled tables |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment
Hosted on **Streamlit Community Cloud** via GitHub repo `USSectorPCA`.
- `requirements.txt` — all dependencies with lower-bound version pins
- `runtime.txt` — specifies `python-3.12` for the cloud build
- No API keys required — `yfinance` uses public Yahoo Finance data

---

## Usage
1. Tick **Use S&P 500 sector preset** (default on) or enter custom tickers
2. Use the **ⓘ** popover next to the preset checkbox to look up ticker names
3. Set date range, return type (log/simple), number of components
4. Configure PCA options: standardise, matrix type (correlation/covariance), Ledoit-Wolf shrinkage
5. Click **Run PCA** — results persist in session state
6. Inspect the side-by-side **Absolute** and **Relative** PCA panels, each with tabs: Scree | Loadings | Biplot | Correlation
7. Use **Constituent Drill-down** to select ETFs and run PCA on their top-10 holdings

---

## Dual PCA — Absolute vs Relative
The app runs two parallel PCAs side by side:

| Mode | Input | What it reveals |
|---|---|---|
| **Absolute** | Raw sector ETF returns | PC1 = broad market factor; subsequent PCs = residual variation |
| **Relative** | Sector return − SPY return (daily) | Market factor removed; PCs reveal rotation: cyclical vs defensive, momentum, etc. |

**Benchmark:** SPY (S&P 500 ETF) — consistent with using ETFs throughout.

### Region Presets
The app supports three region presets selected via a dropdown (defaults to US):

| Region | ETFs | Benchmark | Currency | Sectors |
|---|---|---|---|---|
| US — S&P 500 | XLB/XLC/XLE/XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY | SPY | USD | 11 |
| Europe — STOXX 600 | EXV1-9/EXH1/2/3/4/9 (.DE) | EXSA.DE | EUR | 14 |
| Global — MSCI ACWI | IXP/RXI/KXI/IXC/IXG/IXJ/EXI/MXI/IXN/JXI | ACWI | USD | 10 |

Non-USD regions display a currency notice. All tickers verified to have >1 year of clean data via yfinance as of 2026-04-26.

### S&P 500 Sector Preset (US)
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

## Constituent Drill-down
After running the sector PCA, the user can select one or more ETFs to drill into their top-10 holdings.

**Separate mode** (default): one PCA per ETF, each benchmarked against that ETF's own return. Loadings table shows symbol, name, weight in ETF, PC1, PC2 — sorted by |PC1|.

**Merged mode** (checkbox): all holdings from all selected ETFs are pooled into a single PCA relative to SPY. A Sector column identifies the source ETF for each stock, allowing cross-sector factor structure to be observed.

Both modes display:
- Holdings coverage caption (e.g. "Top 10 holdings represent 61.3% of XLK")
- Explained variance per PC
- Loadings table with colour gradient
- Cumulative PC score projections chart

**Data source:** `yfinance.Ticker.funds_data.top_holdings` — returns up to 10 holdings. Coverage is typically 60–75% of ETF weight, sufficient for PCA signal.

---

## Design Decisions & Notes
- **Adjusted close** prices used throughout to account for splits and dividends.
- **Log returns** default — additive, approximately normal, better suited for PCA.
- **Relative returns** = sector daily return − SPY daily return (not excess over risk-free rate). Pure cross-sectional view, not an alpha measure.
- **Standardise** (default on): scales each return series to unit variance before PCA. When on, covariance of scaled data = correlation matrix, making the matrix-type toggle moot.
- **PCA matrix type** (Correlation / Covariance): only has a distinct effect when standardise is off. The effective matrix used is shown as a UI metric.
- **Ledoit-Wolf shrinkage** (default on): shrinks the sample matrix towards a scaled identity before eigendecomposition. Shrinkage coefficient α shown in UI. With 11 sectors and ~1000 obs, α ≈ 0.01 — becomes more material with shorter windows or larger baskets.
- **PCA implementation**: uses explicit eigendecomposition (`np.linalg.eigh`) rather than sklearn SVD, so the covariance/correlation matrix is always computed and shrinkage can be applied before decomposition.
- **Session state**: all sector PCA results stored in `st.session_state["pca"]`; constituent results in `st.session_state["drill_results"]`. This allows the constituent button to fire independently without re-running the sector PCA.
- **Correlation tab**: each PCA panel (absolute/relative) has a Correlation tab showing the pairwise return correlation heatmap for that input — useful for interpreting loadings (e.g. XLC/XLK correlation collapses from 0.78 absolute to 0.10 relative).

---

## Changelog
| Date | Change |
|---|---|
| 2026-04-25 | Project initialised — CLAUDE.md, requirements, package scaffold |
| 2026-04-25 | Dual PCA (absolute vs relative), SPY benchmark, sector ETF preset, `compute_relative_returns` |
| 2026-04-25 | Ledoit-Wolf shrinkage; explicit correlation matrix eigendecomposition; shrinkage α in UI |
| 2026-04-25 | Standardise toggle and correlation/covariance matrix selector |
| 2026-04-25 | Constituent drill-down: `holdings.py`, per-ETF PCA, PC1/PC2 loadings table |
| 2026-04-25 | Merged constituent mode: pool holdings across ETFs, single PCA vs SPY, Sector column |
| 2026-04-25 | Correlation heatmap moved into Scree/Loadings/Biplot/Correlation tab structure |
| 2026-04-25 | Session state refactor — sector and constituent results persist across button clicks |
| 2026-04-25 | Cumulative PC score projections added to both sector and constituent panels |
| 2026-04-25 | ETF ticker ⓘ popover; scree legend repositioned above chart; sector reference panel removed |
| 2026-04-25 | Weight coverage caption added to constituent results |
| 2026-04-25 | GitHub repo created (USSectorPCA); `runtime.txt` and `.gitignore` added for Streamlit Cloud deployment |
| 2026-04-26 | Region selector added: US (S&P 500), Europe (STOXX 600), Global (MSCI ACWI) — all tickers verified for data quality |
| 2026-04-26 | Global preset replaced: SPDR MSCI World (.DE, EUR, 7 sectors) → iShares MSCI ACWI (USD, 10 sectors, ACWI benchmark) |
| 2026-04-26 | Europe preset fixed and expanded: corrected all ticker→name mappings (were completely scrambled); added EXH1/2/4/9.DE (Oil & Gas, Financial Services, Industrial Goods, Utilities); now 14 sectors |
| 2026-04-26 | Retry logic added to `fetch_adjusted_close` and `fetch_etf_holdings` — handles transient Yahoo Finance rate-limiting on cloud IPs (up to 2 retries, 2s delay) |
| 2026-04-26 | `app.py` re-filters sector tickers against columns surviving `compute_returns`; explicit guard if benchmark is dropped |
