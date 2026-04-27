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
    ├── network.py          # MST construction and Louvain community detection
    └── plots.py            # Plotly charts: scree, loadings heatmap, biplot, correlation heatmap, PC score projections, MST network graph
```

### Key Components
| Module | Responsibility |
|---|---|
| `pca/data.py` | Fetch adjusted close prices via `yfinance` for a list of tickers |
| `pca/returns.py` | Compute log or simple returns; `compute_relative_returns` subtracts benchmark daily return |
| `pca/analysis.py` | Demean, optionally standardise, compute correlation/covariance matrix (with optional LW shrinkage), eigendecompose; returns `PCAResult` dataclass; `run_rolling_pca` for time-series factor analysis |
| `pca/holdings.py` | Fetch ETF top-10 holdings (symbol, name, weight) from `yfinance.Ticker.funds_data` |
| `pca/network.py` | `build_mst()` — Kruskal MST on Mantegna distance (1 − ρ); `detect_communities()` — Louvain on positive-correlation graph via `networkx` |
| `pca/plots.py` | Scree plot, loadings heatmap, biplot, correlation heatmap, cumulative PC score projections, rolling variance chart, rolling loadings heatmap, MST network graph |
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
| `networkx` | MST construction and Louvain community detection |

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
1. Select a **Region** from the dropdown (US / Europe / Global) — defaults to US
2. Use the **ⓘ** popover next to the region dropdown to look up ticker names
3. Tick **Use region preset** (default on) or enter custom tickers
4. Set date range, return type (log/simple), number of components
5. Configure PCA options: standardise, matrix type (correlation/covariance), Ledoit-Wolf shrinkage
6. Click **Run PCA** — results persist in session state
7. Inspect the side-by-side **Absolute** and **Relative** PCA panels, each with tabs: Scree | Loadings | Biplot | Correlation | Network
8. Use **Constituent Drill-down** to select ETFs and run PCA on their top-10 holdings

---

## Dual PCA — Absolute vs Relative
The app runs two parallel PCAs side by side:

| Mode | Input | What it reveals |
|---|---|---|
| **Absolute** | Raw sector ETF returns | PC1 = broad market factor; subsequent PCs = residual variation |
| **Relative** | Sector return − benchmark return (daily) | Market factor removed; PCs reveal rotation: cyclical vs defensive, momentum, etc. |

**Benchmark:** region-dependent — SPY (US), EXSA.DE (Europe), ACWI (Global).

### Region Presets
The app supports four presets selected via a dropdown (defaults to US):

| Region | ETFs | Benchmark | Currency | Sectors |
|---|---|---|---|---|
| US — S&P 500 | XLB/XLC/XLE/XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY | SPY | USD | 11 |
| Europe — STOXX 600 | EXV1-9/EXH1/2/3/4/9 (.DE) | EXSA.DE | EUR | 14 |
| Global — MSCI ACWI | IXP/RXI/KXI/IXC/IXG/IXJ/EXI/MXI/IXN/JXI | ACWI | USD | 10 |
| US — Style Factors | IWF/VTV/IWM/USMV/MTUM/QUAL/VLUE/SIZE | SPY | USD | 8 |

Non-USD regions display a currency notice. All tickers verified to have >1 year of clean data via yfinance as of 2026-04-26.

### Style Factors Preset
| Ticker | Factor | History from |
|---|---|---|
| IWF | Growth | 2000 |
| VTV | Value (Broad) | 2004 |
| IWM | Small Cap | 2000 |
| USMV | Min Volatility | **2011-10-20** |
| MTUM | Momentum | **2013-04-18** |
| QUAL | Quality | **2013-07-18** |
| VLUE | Value (Factor) | **2013-04-18** |
| SIZE | Size | **2013-04-18** |

In relative space (vs SPY), PC1 typically captures the value/growth rotation; PC2 captures the risk-on/risk-off (small cap vs defensive) dynamic.

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

**Merged mode** (checkbox): all holdings from all selected ETFs are pooled into a single PCA relative to the region benchmark. A Sector column identifies the source ETF for each stock, allowing cross-sector factor structure to be observed.

Both modes display:
- Holdings coverage caption (e.g. "Top 10 holdings represent 61.3% of XLK")
- Period statistics table with benchmark row at top (bold) — total return, annualised return, volatility, beta, risk-adjusted return
- Loadings table with colour gradient, explained variance per PC
- MST network graph of constituent correlations with Louvain community colouring
- Cumulative PC score projections chart
- Rolling PCA: explained variance and PC1 loadings heatmap using the same window as the sector rolling PCA, with disclaimer noting static constituent assumption

**Data source:** `yfinance.Ticker.funds_data.top_holdings` — returns up to 10 holdings. Coverage is typically 60–75% of ETF weight, sufficient for PCA signal.

**Benchmark choice (separate mode):** user can benchmark each ETF's constituents against the sector ETF itself (reveals within-sector sub-factors) or against the region benchmark (shows which constituents lead/lag the market).

---

## Design Decisions & Notes
- **Adjusted close** prices used throughout to account for splits and dividends.
- **Log returns** default — additive, approximately normal, better suited for PCA.
- **Relative returns** = sector daily return − benchmark daily return (not excess over risk-free rate). Pure cross-sectional view, not an alpha measure. Benchmark is region-dependent.
- **Standardise** (default on): scales each return series to unit variance before PCA. When on, covariance of scaled data = correlation matrix, making the matrix-type toggle moot.
- **PCA matrix type** (Correlation / Covariance): only has a distinct effect when standardise is off. The effective matrix used is shown as a UI metric.
- **Ledoit-Wolf shrinkage** (default on): shrinks the sample matrix towards a scaled identity before eigendecomposition. Shrinkage coefficient α shown in UI. With 11 sectors and ~1000 obs, α ≈ 0.01 — becomes more material with shorter windows or larger baskets.
- **PCA implementation**: uses explicit eigendecomposition (`np.linalg.eigh`) rather than sklearn SVD, so the covariance/correlation matrix is always computed and shrinkage can be applied before decomposition.
- **Session state**: all sector PCA results stored in `st.session_state["pca"]`; constituent results in `st.session_state["drill_results"]`. This allows the constituent button to fire independently without re-running the sector PCA.
- **Correlation tab**: each PCA panel (absolute/relative) has a Correlation tab showing the pairwise return correlation heatmap for that input — useful for interpreting loadings (e.g. XLC/XLK correlation collapses from 0.78 absolute to 0.10 relative).
- **`fill_limit=3`** in `compute_returns` for constituent PCA: forward-fills prices up to 3 days before computing returns, bridging different exchange holiday calendars (e.g. UK/EU/Nordic stocks in European ETF holdings).
- **Constituent diagnostics**: when tickers are dropped (no price data or >5% missing), the app names them explicitly so the user can distinguish transient rate-limiting from structural unavailability.
- **Holdings retry**: `fetch_etf_holdings` retries up to 4 times with exponential backoff (3s, 6s, 9s, 12s) — necessary for European .DE ETFs which are rate-limited more aggressively on cloud IPs.
- **Sector ticker re-filtering**: after `compute_returns`, sector tickers are re-checked against surviving columns; benchmark absence triggers an explicit error and `st.stop()`.
- **Rolling PCA** (`run_rolling_pca` in `analysis.py`): runs PCA on a rolling window (63/126/252 days) and returns time series of explained variance ratios and loadings per PC. Sign ambiguity resolved by aligning each window to the previous via dot product sign; first window uses sklearn convention (largest absolute loading is positive). Displayed after the dual PCA section and also per-ETF in constituent drill-down. If the selected window exceeds the available trading days, a descriptive info message is shown instead of crashing.
- **Period statistics benchmark row**: `_with_benchmark_row()` prepends the benchmark's own stats (beta=1 by definition) as the first row, bolded, in both sector and constituent period statistics tables.
- **Style factors preset**: relative PCA vs SPY on factor ETFs (Growth, Value, Momentum, Quality, Min Vol, Size, Small Cap) isolates factor rotation dynamics without the market return.
- **MST network graph** (`pca/network.py`, `plots.mst_plot`): Minimum Spanning Tree of the pairwise return correlation matrix using Mantegna distance (1 − ρ). Node layout computed via Kamada-Kawai on the full distance matrix so geometrically close nodes are highly correlated. Nodes coloured by Louvain community (detected on the positive-correlation graph using `networkx.community.louvain_communities`). Edge thickness scales with |correlation|; edge hover shows exact correlation value. Appears as a **Network** tab in both Absolute and Relative PCA panels, and inline in constituent drill-down results (after the loadings table, before PC score projections). The relative-returns MST is particularly interpretable — with the market factor stripped, topology reflects true rotation dynamics rather than beta clustering.

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
| 2026-04-26 | Retry logic added to `fetch_adjusted_close` and `fetch_etf_holdings` — handles transient Yahoo Finance rate-limiting on cloud IPs |
| 2026-04-26 | `app.py` re-filters sector tickers against columns surviving `compute_returns`; explicit guard if benchmark is dropped |
| 2026-04-26 | `compute_returns` gains `fill_limit` param; constituent PCA uses `fill_limit=3` to handle mixed exchange calendars |
| 2026-04-26 | Constituent diagnostics: names tickers with no price data or dropped by >5% missing threshold |
| 2026-04-26 | History warnings added for Global preset: ACWI benchmark (2008-03-28); benchmark vs sector warning distinction |
| 2026-04-26 | `fetch_etf_holdings` retry increased to 4 attempts with exponential backoff for European ETF rate-limiting |
| 2026-04-26 | Rolling PCA added: `run_rolling_pca()` in analysis.py; rolling variance and loadings heatmap charts in plots.py; Rolling PCA section in app.py (window 1Q/6M/1Y, PC1/PC2 toggle, abs vs rel side by side) |
| 2026-04-26 | Rolling PCA added to constituent drill-down results with static-holdings disclaimer |
| 2026-04-26 | US — Style Factors preset added: IWF/VTV/IWM/USMV/MTUM/QUAL/VLUE/SIZE vs SPY |
| 2026-04-26 | Benchmark row added to period statistics tables (bold, beta=1, top row) for both sector and constituent views |
| 2026-04-27 | MST network graph added: `pca/network.py` (`build_mst`, `detect_communities`); `plots.mst_plot`; Network tab in sector PCA panels; inline in constituent drill-down after loadings table |
| 2026-04-27 | Rolling PCA window guard: shows info message instead of `KeyError` when selected window exceeds available trading days |
