import streamlit as st
from datetime import date, timedelta

import pandas as pd

from pca.data import fetch_adjusted_close
from pca.returns import compute_returns, compute_relative_returns, compute_period_stats
from pca.analysis import run_pca, run_rolling_pca, PCAResult
from pca.holdings import fetch_etf_holdings
from pca import plots

st.set_page_config(page_title="PCA Equity Analysis", layout="wide")

# --- Region definitions ---
REGIONS = {
    "US — S&P 500 Sectors": {
        "etfs": ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
        "benchmark": "SPY",
        "benchmark_name": "S&P 500",
        "currency": "USD",
        "names": {
            "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy",
            "XLF": "Financials", "XLI": "Industrials", "XLK": "Information Technology",
            "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
            "XLV": "Health Care", "XLY": "Consumer Discretionary",
        },
        "history_warnings": {
            "XLC": (date(2018, 6, 19), "Communication Services"),
            "XLRE": (date(2015, 10, 7), "Real Estate"),
        },
    },
    "Europe — STOXX 600 Sectors": {
        "etfs": ["EXV1.DE", "EXV2.DE", "EXV3.DE", "EXV4.DE", "EXV5.DE",
                 "EXV6.DE", "EXV7.DE", "EXV8.DE", "EXH1.DE", "EXH2.DE",
                 "EXH3.DE", "EXH4.DE", "EXH9.DE", "EXV9.DE"],
        "benchmark": "EXSA.DE",
        "benchmark_name": "STOXX Europe 600",
        "currency": "EUR",
        "names": {
            "EXV1.DE": "Banks",                    "EXV2.DE": "Telecommunications",
            "EXV3.DE": "Technology",               "EXV4.DE": "Health Care",
            "EXV5.DE": "Automobiles & Parts",      "EXV6.DE": "Basic Resources",
            "EXV7.DE": "Chemicals",                "EXV8.DE": "Construction & Materials",
            "EXH1.DE": "Oil & Gas",                "EXH2.DE": "Financial Services",
            "EXH3.DE": "Food & Beverage",          "EXH4.DE": "Industrial Goods & Services",
            "EXH9.DE": "Utilities",                "EXV9.DE": "Travel & Leisure",
        },
        "history_warnings": {},
    },
    "Global — MSCI ACWI Sectors": {
        "etfs": ["IXP", "RXI", "KXI", "IXC", "IXG", "IXJ", "EXI", "MXI", "IXN", "JXI"],
        "benchmark": "ACWI",
        "benchmark_name": "MSCI ACWI",
        "currency": "USD",
        "names": {
            "IXP": "Communication Services", "RXI": "Consumer Discretionary",
            "KXI": "Consumer Staples",        "IXC": "Energy",
            "IXG": "Financials",              "IXJ": "Health Care",
            "EXI": "Industrials",             "MXI": "Materials",
            "IXN": "Technology",              "JXI": "Utilities",
        },
        "history_warnings": {
            "ACWI": (date(2008, 3, 28), "MSCI ACWI benchmark"),
            "RXI":  (date(2006, 9, 21), "Consumer Discretionary"),
            "KXI":  (date(2006, 9, 22), "Consumer Staples"),
            "EXI":  (date(2006, 9, 27), "Industrials"),
            "MXI":  (date(2006, 9, 22), "Materials"),
            "JXI":  (date(2006, 9, 22), "Utilities"),
        },
    },
    "US — Style Factors": {
        "etfs": ["IWF", "VTV", "IWM", "USMV", "MTUM", "QUAL", "VLUE", "SIZE"],
        "benchmark": "SPY",
        "benchmark_name": "S&P 500",
        "currency": "USD",
        "names": {
            "IWF":  "Growth",
            "VTV":  "Value (Broad)",
            "IWM":  "Small Cap",
            "USMV": "Min Volatility",
            "MTUM": "Momentum",
            "QUAL": "Quality",
            "VLUE": "Value (Factor)",
            "SIZE": "Size",
        },
        "history_warnings": {
            "USMV": (date(2011, 10, 20), "Min Volatility"),
            "MTUM": (date(2013, 4, 18), "Momentum"),
            "QUAL": (date(2013, 7, 18), "Quality"),
            "VLUE": (date(2013, 4, 18), "Value (Factor)"),
            "SIZE": (date(2013, 4, 18), "Size"),
        },
    },
}

# --- Sidebar controls ---
with st.sidebar:
    st.header("Configuration")

    region_col, info_col = st.columns([5, 1])
    region_key = region_col.selectbox("Region", list(REGIONS.keys()), index=0)
    region = REGIONS[region_key]
    SECTOR_NAMES = region["names"]
    BENCHMARK    = region["benchmark"]

    with info_col.popover("ⓘ"):
        st.markdown(f"**{region_key}**")
        for ticker, name in SECTOR_NAMES.items():
            st.markdown(f"**{ticker}** — {name}")
        st.markdown(f"**{BENCHMARK}** — {region['benchmark_name']} (benchmark)")
        if region["currency"] != "USD":
            st.caption(f"All returns in {region['currency']}.")

    use_preset = st.checkbox("Use region preset", value=True)
    if use_preset:
        st.text_area("Tickers (read-only)", value=", ".join(region["etfs"]), disabled=True, height=80)
        st.caption(f"Benchmark: {BENCHMARK} ({region['benchmark_name']}) · {region['currency']}")
        tickers = region["etfs"][:]
    else:
        ticker_input = st.text_area(
            "Tickers (comma-separated)",
            value=", ".join(region["etfs"]),
            height=80,
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", value=date.today() - timedelta(days=5 * 365))
    end_date = col2.date_input("End date", value=date.today())

    return_type = st.radio("Return type", ["log", "simple"], horizontal=True)
    n_components = st.slider("Number of components", min_value=2, max_value=10, value=5)

    st.divider()
    standardize = st.checkbox(
        "Standardise returns",
        value=True,
        help=(
            "Scale each return series to zero mean and unit variance before PCA. "
            "Prevents high-volatility sectors dominating the components. "
            "When on, the covariance of the scaled data equals the correlation matrix."
        ),
    )
    matrix_type = st.radio(
        "PCA matrix",
        ["Correlation", "Covariance"],
        horizontal=True,
        help=(
            "Matrix to eigendecompose. When 'Standardise returns' is on, both choices "
            "are equivalent — the effective matrix will always be Correlation."
        ),
    )
    if standardize and matrix_type == "Covariance":
        st.caption("ℹ️ Standardised returns → covariance = correlation. Effective matrix: **Correlation**.")

    use_shrinkage = st.checkbox(
        "Ledoit-Wolf covariance shrinkage",
        value=True,
        help=(
            "Shrinks the sample matrix towards a scaled identity before "
            "eigendecomposition. Improves eigenvector stability when the number of "
            "assets is not much smaller than the number of observations."
        ),
    )

    run = st.button("Run PCA", type="primary", use_container_width=True)

# --- Helpers ---

def _style_stats(df: "pd.DataFrame") -> "pd.io.formats.style.Styler":
    """Format and heatmap a period statistics DataFrame.

    Expects columns: ticker, name, beta, total_return, ann_return, ann_volatility, risk_adjusted.
    Ann. Return and Risk-Adjusted are heatmapped with a diverging blue/red scale centred at zero.
    """
    display = df.rename(columns={
        "beta":           "Beta",
        "total_return":   "Total Return",
        "ann_return":     "Ann. Return",
        "ann_volatility": "Ann. Volatility",
        "risk_adjusted":  "Risk-Adjusted",
    })

    def _diverging(styler, col):
        if col not in display.columns:
            return styler
        abs_max = display[col].abs().max()
        if abs_max == 0 or pd.isna(abs_max):
            return styler
        return styler.background_gradient(
            subset=[col], cmap="RdBu", vmin=-abs_max, vmax=abs_max
        )

    fmt = {
        "Total Return":    "{:+.2%}",
        "Ann. Return":     "{:+.2%}",
        "Ann. Volatility": "{:.2%}",
        "Risk-Adjusted":   "{:+.2f}",
    }
    if "Beta" in display.columns:
        fmt["Beta"] = "{:.2f}"

    styler = display.style.format(fmt)
    styler = _diverging(styler, "Ann. Return")
    styler = _diverging(styler, "Risk-Adjusted")
    return styler


def _pca_tabs(label: str, result: PCAResult, n_comp: int, returns: "pd.DataFrame") -> None:
    tab1, tab2, tab3, tab4 = st.tabs(["Scree", "Loadings", "Biplot", "Correlation"])
    with tab1:
        st.plotly_chart(plots.scree_plot(result), use_container_width=True)
    with tab2:
        st.plotly_chart(plots.loadings_heatmap(result), use_container_width=True)
        with st.expander("Loadings table"):
            loadings_display = result.loadings.copy()
            loadings_display.insert(0, "name", loadings_display.index.map(SECTOR_NAMES).fillna(""))
            st.dataframe(
                loadings_display.style.background_gradient(
                    subset=result.loadings.columns.tolist(), cmap="RdBu_r", axis=None
                ),
                use_container_width=True,
            )
    with tab3:
        pc_opts = [f"PC{i+1}" for i in range(n_comp)]
        c1, c2 = st.columns(2)
        px_sel = c1.selectbox("X axis", pc_opts, index=0, key=f"{label}_x")
        py_sel = c2.selectbox("Y axis", pc_opts, index=1, key=f"{label}_y")
        pc_x, pc_y = int(px_sel[2:]), int(py_sel[2:])
        if pc_x == pc_y:
            st.warning("Select two different components.")
        else:
            st.plotly_chart(plots.biplot(result, pc_x, pc_y), use_container_width=True)
    with tab4:
        st.plotly_chart(plots.correlation_heatmap(returns), use_container_width=True)

    st.subheader("PC Score Projections")
    st.caption(
        "Cumulative sum of daily factor scores over the selected horizon. "
        "Each panel shows how one latent factor has trended — a rising line means "
        "that factor's influence accumulated positively over the period."
    )
    st.plotly_chart(plots.pc_scores_chart(result), use_container_width=True)


# --- Main panel ---
if run:
    if len(tickers) < 3:
        st.error("Please enter at least 3 tickers.")
        st.stop()

    if use_preset:
        for ticker, (cutoff, label) in region["history_warnings"].items():
            if start_date < cutoff:
                if ticker == BENCHMARK:
                    st.error(
                        f"Benchmark {ticker} ({label}) only has data from {cutoff}. "
                        "Please set your start date to on or after this date."
                    )
                else:
                    st.warning(
                        f"{ticker} ({label}) only has data from {cutoff}. "
                        "It will be included from that date; earlier rows will be dropped."
                    )

    fetch_tickers = list(dict.fromkeys(tickers + [BENCHMARK]))

    with st.spinner("Fetching price data…"):
        prices = fetch_adjusted_close(fetch_tickers, str(start_date), str(end_date))

    available = list(prices.columns)
    if BENCHMARK not in available:
        st.error(f"Could not retrieve benchmark ({BENCHMARK}) data. Cannot continue.")
        st.stop()

    missing = [t for t in tickers if t not in available]
    if missing:
        st.warning(f"No data found for: {', '.join(missing)}")

    sector_tickers = [t for t in tickers if t in available]
    if len(sector_tickers) < 3:
        st.error("Fewer than 3 sector tickers have data. Adjust your selection.")
        st.stop()

    all_returns = compute_returns(prices[sector_tickers + [BENCHMARK]], method=return_type)
    if BENCHMARK not in all_returns.columns:
        st.error(f"Benchmark {BENCHMARK} has insufficient data for the selected period.")
        st.stop()
    benchmark_series = all_returns[BENCHMARK]
    dropped = [t for t in sector_tickers if t not in all_returns.columns]
    if dropped:
        st.warning(f"Dropped due to insufficient data after return computation: {', '.join(dropped)}")
    sector_tickers = [t for t in sector_tickers if t in all_returns.columns]
    sector_returns = all_returns[sector_tickers]
    relative_returns = compute_relative_returns(sector_returns, benchmark_series)

    n_comp = min(n_components, len(sector_tickers))
    pca_kwargs = dict(
        n_components=n_comp,
        use_shrinkage=use_shrinkage,
        standardize=standardize,
        matrix_type=matrix_type.lower(),
    )
    result_abs = run_pca(sector_returns, **pca_kwargs)
    result_rel = run_pca(relative_returns, **pca_kwargs)

    # Persist results so constituent button reruns don't wipe the page
    st.session_state["pca"] = dict(
        sector_returns=sector_returns,
        relative_returns=relative_returns,
        benchmark_series=benchmark_series,
        result_abs=result_abs,
        result_rel=result_rel,
        sector_tickers=sector_tickers,
        n_comp=n_comp,
        pca_kwargs=pca_kwargs,
        start_date=start_date,
        end_date=end_date,
        return_type=return_type,
        region_key=region_key,
        sector_names=SECTOR_NAMES,
        benchmark_name=region["benchmark_name"],
        currency=region["currency"],
    )
    st.session_state.pop("drill_results", None)   # clear stale drill-down on fresh run

if "pca" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run PCA**.")
    st.stop()

# --- Unpack persisted state ---
_s = st.session_state["pca"]
sector_returns   = _s["sector_returns"]
relative_returns = _s["relative_returns"]
benchmark_series = _s["benchmark_series"]
result_abs       = _s["result_abs"]
result_rel       = _s["result_rel"]
sector_tickers   = _s["sector_tickers"]
n_comp           = _s["n_comp"]
pca_kwargs       = _s["pca_kwargs"]
start_date       = _s["start_date"]
end_date         = _s["end_date"]
return_type      = _s["return_type"]
_region_key      = _s["region_key"]
SECTOR_NAMES     = _s["sector_names"]
BENCHMARK        = benchmark_series.name          # ticker symbol e.g. "SPY", "EXSA.DE"
_benchmark_name  = _s.get("benchmark_name", BENCHMARK)
_currency        = _s.get("currency", "USD")

st.title(f"Equity Return PCA — {_region_key}")
if _currency != "USD":
    st.caption(f"All returns denominated in {_currency}.")

# --- Summary ---
st.subheader("Data Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Sectors used", len(sector_tickers))
m2.metric("Trading days", sector_returns.shape[0])
m3.metric("Date range", f"{sector_returns.index[0].date()} → {sector_returns.index[-1].date()}")

with st.expander("Period Statistics"):
    stats = compute_period_stats(sector_returns, method=return_type, benchmark=benchmark_series)
    stats_df = stats.reset_index()
    stats_df.columns = ["ticker", "beta", "total_return", "ann_return", "ann_volatility", "risk_adjusted"]
    stats_df.insert(1, "name", stats_df["ticker"].map(SECTOR_NAMES).fillna(""))
    st.dataframe(_style_stats(stats_df), use_container_width=True, hide_index=True)

st.divider()

# --- Dual PCA columns ---
col_abs, col_rel = st.columns(2)

with col_abs:
    st.subheader("Absolute Returns PCA")
    st.caption("PCA on raw sector returns — PC1 typically captures the broad market factor. PC2 is often Risk On/Risk Off.")
    n_met = 4 if result_abs.shrinkage_coefficient is not None else 3
    a_cols = st.columns(n_met)
    a_cols[0].metric("Variance explained (all PCs)", f"{result_abs.cumulative_variance[-1]*100:.1f}%")
    a_cols[1].metric("PC1 explains", f"{result_abs.explained_variance_ratio[0]*100:.1f}%")
    a_cols[2].metric("Matrix", result_abs.matrix_type.capitalize())
    if result_abs.shrinkage_coefficient is not None:
        a_cols[3].metric("LW shrinkage α", f"{result_abs.shrinkage_coefficient:.3f}")
    _pca_tabs("abs", result_abs, n_comp, sector_returns)

with col_rel:
    st.subheader("Relative Returns PCA")
    st.caption(f"PCA on sector returns minus {_benchmark_name} — isolates rotation dynamics beyond the market.")
    n_met = 4 if result_rel.shrinkage_coefficient is not None else 3
    r_cols = st.columns(n_met)
    r_cols[0].metric("Variance explained (all PCs)", f"{result_rel.cumulative_variance[-1]*100:.1f}%")
    r_cols[1].metric("PC1 explains", f"{result_rel.explained_variance_ratio[0]*100:.1f}%")
    r_cols[2].metric("Matrix", result_rel.matrix_type.capitalize())
    if result_rel.shrinkage_coefficient is not None:
        r_cols[3].metric("LW shrinkage α", f"{result_rel.shrinkage_coefficient:.3f}")
    _pca_tabs("rel", result_rel, n_comp, relative_returns)

# --- Rolling PCA ---
st.divider()
st.subheader("Rolling PCA")
st.caption("How factor structure and sector exposures evolve through time.")

_min_window = 63
if sector_returns.shape[0] < _min_window:
    st.info("Not enough data for rolling PCA — select a longer date range.")
else:
    _r_col1, _r_col2, _r_col3 = st.columns([2, 2, 3])
    roll_window = _r_col1.select_slider(
        "Rolling window",
        options=[63, 126, 252],
        value=252,
        format_func=lambda v: {63: "1 quarter", 126: "6 months", 252: "1 year"}[v],
    )
    roll_pc = _r_col2.radio("Loadings to display", ["PC1", "PC2"], horizontal=True)
    _r_col3.caption(
        "**Explained variance** — how much of total cross-sectional variation each PC captures over time.  \n"
        "**Loadings** — which sectors drive each factor and how that changes."
    )

    with st.spinner("Computing rolling PCA…"):
        _roll_kwargs = dict(
            window=roll_window,
            n_components=min(3, n_comp),
            use_shrinkage=pca_kwargs["use_shrinkage"],
            standardize=pca_kwargs["standardize"],
            matrix_type=pca_kwargs["matrix_type"],
        )
        _ev_abs, _ld_abs = run_rolling_pca(sector_returns, **_roll_kwargs)
        _ev_rel, _ld_rel = run_rolling_pca(relative_returns, **_roll_kwargs)

    # Rename ticker columns to sector names for readability
    def _rename_cols(df):
        return df.rename(columns=SECTOR_NAMES) if not df.empty else df

    _rc_abs, _rc_rel = st.columns(2)
    with _rc_abs:
        st.markdown("**Absolute Returns**")
        st.plotly_chart(
            plots.rolling_variance_chart(_ev_abs, "Rolling Explained Variance — Absolute"),
            use_container_width=True,
        )
        st.plotly_chart(
            plots.rolling_loadings_heatmap(
                _rename_cols(_ld_abs[roll_pc]),
                f"Rolling {roll_pc} Loadings — Absolute",
            ),
            use_container_width=True,
        )
    with _rc_rel:
        st.markdown(f"**Relative Returns (vs {_benchmark_name})**")
        st.plotly_chart(
            plots.rolling_variance_chart(_ev_rel, "Rolling Explained Variance — Relative"),
            use_container_width=True,
        )
        st.plotly_chart(
            plots.rolling_loadings_heatmap(
                _rename_cols(_ld_rel[roll_pc]),
                f"Rolling {roll_pc} Loadings — Relative",
            ),
            use_container_width=True,
        )

# --- Constituent drill-down ---
st.divider()
st.subheader("Constituent Drill-down")

drill_etfs = st.multiselect(
    "ETFs to drill into",
    options=sector_tickers,
    format_func=lambda t: f"{t} — {SECTOR_NAMES.get(t, t)}",
    default=[],
)

merge_constituents = st.checkbox(
    "Merge constituents across selected ETFs",
    value=False,
    help=(
        "When checked, all holdings from the selected ETFs are pooled into a single PCA "
        f"relative to {_benchmark_name} — useful for seeing how stocks from different sectors "
        "interleave in factor space. When unchecked, a separate PCA is run for each ETF."
    ),
)

if merge_constituents:
    constituent_benchmark = "market"   # sentinel — always vs market in merged mode
    st.caption(
        f"Merged mode: single PCA across all selected ETF constituents, relative to "
        f"{_benchmark_name} ({BENCHMARK}). Sector column identifies the source ETF."
    )
else:
    constituent_benchmark = st.radio(
        "Constituent benchmark",
        ["vs sector ETF", f"vs {_benchmark_name}"],
        index=0,
        horizontal=True,
        help=(
            "**vs sector ETF**: strips the sector's own return to reveal within-sector "
            "sub-factors (e.g. semis vs software within XLK).\n\n"
            f"**vs {_benchmark_name}**: strips the market return to show which constituents "
            "are leading or lagging the index."
        ),
    )
    if constituent_benchmark == "vs sector ETF":
        st.caption("Separate mode: one PCA per ETF, each relative to that ETF's own return.")
    else:
        st.caption(f"Separate mode: one PCA per ETF, each relative to {_benchmark_name} ({BENCHMARK}).")

def _build_loadings_table(c_result, holdings_df, n_c_comp):
    """Join PCA loadings with holdings metadata and sort by |PC1|."""
    pc_cols = [f"PC{i+1}" for i in range(n_c_comp)]
    df = c_result.loadings.reset_index().rename(columns={"index": "symbol"})
    df.columns = ["symbol"] + pc_cols
    df = df.merge(holdings_df, on="symbol", how="left")
    df["name"]   = df["name"].fillna("")
    df["weight"] = df["weight"].fillna(0.0)
    return df.sort_values("PC1", key=abs, ascending=False)

def _render_loadings_table(loadings_df, c_result, n_c_comp, extra_cols=None):
    pc_cols = [f"PC{i+1}" for i in range(n_c_comp)]
    var_labels = "  |  ".join(
        f"PC{i+1}: {c_result.explained_variance_ratio[i]*100:.1f}%"
        for i in range(n_c_comp)
    )
    st.caption(f"Explained variance — {var_labels}  |  Sorted by |PC1|")
    base_cols = ["symbol", "name"] + (extra_cols or []) + ["weight"] + pc_cols
    display = loadings_df[base_cols].rename(columns={"weight": "Weight %"})
    fmt = {"Weight %": "{:.1f}%"}
    fmt.update({c: "{:+.3f}" for c in pc_cols})
    styled = (
        display.style
        .format(fmt)
        .background_gradient(subset=pc_cols, cmap="RdBu_r", vmin=-1, vmax=1)
        .set_properties(**{"text-align": "left"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

if drill_etfs and st.button("Run Constituent PCA", type="secondary"):
    # Fetch all holdings up front
    all_holdings = {}
    for etf in drill_etfs:
        with st.spinner(f"Fetching {etf} holdings…"):
            h = fetch_etf_holdings(etf)
        if h.empty:
            st.warning(f"Could not retrieve holdings for {etf}.")
        else:
            h["etf"] = etf
            h["sector"] = SECTOR_NAMES.get(etf, etf)
            all_holdings[etf] = h

    if not all_holdings:
        st.session_state.pop("drill_results", None)
    elif merge_constituents:
        # Pool all unique stocks; first-seen ETF wins on duplicates
        combined = pd.concat(all_holdings.values(), ignore_index=True)
        combined = combined.drop_duplicates(subset="symbol", keep="first")

        all_syms = combined["symbol"].tolist()
        fetch_list = list(dict.fromkeys(all_syms + [BENCHMARK]))

        with st.spinner(f"Fetching price data for {len(all_syms)} stocks…"):
            c_prices = fetch_adjusted_close(fetch_list, str(start_date), str(end_date))

        bm_ticker = BENCHMARK
        available = [t for t in all_syms if t in c_prices.columns]
        no_price = [t for t in all_syms if t not in c_prices.columns]
        bm_in_prices = bm_ticker in c_prices.columns

        if not bm_in_prices:
            st.warning("Could not fetch benchmark for benchmarking.")
            st.session_state.pop("drill_results", None)
        elif len(available) < 3:
            st.warning(f"Fewer than 3 stocks have price data. No data: {', '.join(no_price) if no_price else 'none fetched'}.")
            st.session_state.pop("drill_results", None)
        else:
            if no_price:
                st.info(f"No price data fetched for: {', '.join(no_price)} — excluded.")
            bm_col = bm_ticker
            c_all_rets = compute_returns(c_prices[available + [bm_col]], method=return_type, fill_limit=3)
            # Re-filter: compute_returns may drop tickers with too many missing values
            dropped_clean = [t for t in available if t not in c_all_rets.columns]
            available = [t for t in available if t in c_all_rets.columns]
            if dropped_clean:
                st.info(f"Dropped after return cleaning (>5% missing): {', '.join(dropped_clean)}.")
            if len(available) < 3:
                st.warning("Fewer than 3 stocks have sufficient data after cleaning.")
                st.session_state.pop("drill_results", None)
            else:
                c_rel_rets = compute_relative_returns(c_all_rets[available], c_all_rets[bm_col])
                n_c_comp = min(2, len(available))
                c_result = run_pca(c_rel_rets, **{**pca_kwargs, "n_components": n_c_comp})
                loadings_df = _build_loadings_table(
                    c_result,
                    combined[["symbol", "name", "sector", "weight"]],
                    n_c_comp,
                )
                weight_coverage = (
                    combined.groupby("sector")["weight"].sum()
                    .rename(index=lambda s: next((e for e, n in SECTOR_NAMES.items() if n == s), s))
                    .to_dict()
                )
                st.session_state["drill_results"] = {
                    "_merged": {
                        "result": c_result,
                        "loadings_df": loadings_df,
                        "n_c_comp": n_c_comp,
                        "merged": True,
                        "etf_labels": ", ".join(drill_etfs),
                        "weight_coverage": weight_coverage,
                        "period_stats": compute_period_stats(c_rel_rets, method=return_type, benchmark=c_all_rets[bm_col]),
                        "rel_rets": c_rel_rets,
                    }
                }
    else:
        drill_results = {}
        use_market_bm = constituent_benchmark != "vs sector ETF"

        for etf, holdings in all_holdings.items():
            constituent_tickers = holdings["symbol"].tolist()
            bm_ticker = BENCHMARK if use_market_bm else etf
            fetch_list = list(dict.fromkeys(constituent_tickers + [etf, BENCHMARK] if use_market_bm else constituent_tickers + [etf]))

            with st.spinner(f"Fetching price data for {etf} constituents…"):
                c_prices = fetch_adjusted_close(fetch_list, str(start_date), str(end_date))

            available_constituents = [t for t in constituent_tickers if t in c_prices.columns]
            no_price = [t for t in constituent_tickers if t not in c_prices.columns]
            if bm_ticker not in c_prices.columns:
                drill_results[etf] = {"error": f"Could not retrieve benchmark ({bm_ticker}) for {etf}."}
                continue
            if len(available_constituents) < 3:
                detail = f" (no price data: {', '.join(no_price)})" if no_price else ""
                drill_results[etf] = {"error": f"Fewer than 3 constituents have data for {etf}{detail}."}
                continue

            price_cols = list(dict.fromkeys(available_constituents + [bm_ticker]))
            c_all_rets = compute_returns(c_prices[price_cols], method=return_type, fill_limit=3)
            # Re-filter after compute_returns — it may drop tickers with too many missing values
            dropped_clean = [t for t in available_constituents if t not in c_all_rets.columns]
            available_constituents = [t for t in available_constituents if t in c_all_rets.columns]
            if len(available_constituents) < 3:
                detail = ""
                if no_price:
                    detail += f" No price data: {', '.join(no_price)}."
                if dropped_clean:
                    detail += f" Dropped (>5% missing): {', '.join(dropped_clean)}."
                drill_results[etf] = {"error": f"Fewer than 3 constituents have sufficient data for {etf} after cleaning.{detail}"}
                continue

            c_rel_rets = compute_relative_returns(c_all_rets[available_constituents], c_all_rets[bm_ticker])

            n_c_comp = min(2, len(available_constituents))
            c_result = run_pca(c_rel_rets, **{**pca_kwargs, "n_components": n_c_comp})

            loadings_df = _build_loadings_table(
                c_result, holdings[["symbol", "name", "weight"]], n_c_comp
            )
            drill_results[etf] = {
                "result": c_result,
                "loadings_df": loadings_df,
                "n_c_comp": n_c_comp,
                "merged": False,
                "weight_coverage": {etf: holdings["weight"].sum()},
                "period_stats": compute_period_stats(c_rel_rets, method=return_type, benchmark=c_all_rets[bm_ticker]),
                "bm_label": _benchmark_name if use_market_bm else etf,
                "rel_rets": c_rel_rets,
            }

        st.session_state["drill_results"] = drill_results

# Render persisted drill-down results
if "drill_results" in st.session_state:
    for key, data in st.session_state["drill_results"].items():
        if "error" in data:
            st.warning(data["error"])
            continue

        coverage = data.get("weight_coverage", {})

        if data.get("merged"):
            st.markdown(f"#### Merged: {data['etf_labels']} — constituents vs {BENCHMARK}")
            cov_parts = [f"{etf}: {w:.1f}%" for etf, w in coverage.items()]
            st.caption(f"Top 10 holdings coverage — {' | '.join(cov_parts)}")
        else:
            bm_label = data.get("bm_label", key)
            st.markdown(f"#### {key} — {SECTOR_NAMES.get(key, key)}  ·  vs {bm_label}")
            total_w = coverage.get(key, 0.0)
            st.caption(f"Top 10 holdings represent **{total_w:.1f}%** of {key}")

        with st.expander("Period Statistics"):
            c_stats = data["period_stats"]
            name_map = data["loadings_df"].set_index("symbol")["name"].to_dict()
            c_stats_df = c_stats.reset_index()
            c_stats_df.columns = ["ticker", "beta", "total_return", "ann_return", "ann_volatility", "risk_adjusted"]
            c_stats_df.insert(1, "name", c_stats_df["ticker"].map(name_map).fillna(""))
            st.dataframe(_style_stats(c_stats_df), use_container_width=True, hide_index=True)

        _render_loadings_table(data["loadings_df"], data["result"], data["n_c_comp"], extra_cols=["sector"] if data.get("merged") else None)

        st.plotly_chart(plots.pc_scores_chart(data["result"]), use_container_width=True)

        # Rolling PCA for constituents
        _c_rets = data.get("rel_rets")
        if _c_rets is not None and len(_c_rets) >= roll_window:
            _c_name_map = data["loadings_df"].set_index("symbol")["name"].to_dict()
            _c_ev, _c_ld = run_rolling_pca(
                _c_rets,
                window=roll_window,
                n_components=min(2, data["n_c_comp"]),
                use_shrinkage=pca_kwargs["use_shrinkage"],
                standardize=pca_kwargs["standardize"],
                matrix_type=pca_kwargs["matrix_type"],
            )
            _cc1, _cc2 = st.columns(2)
            with _cc1:
                st.plotly_chart(
                    plots.rolling_variance_chart(_c_ev, "Rolling Explained Variance"),
                    use_container_width=True,
                )
            with _cc2:
                st.plotly_chart(
                    plots.rolling_loadings_heatmap(
                        _c_ld["PC1"].rename(columns=_c_name_map),
                        "Rolling PC1 Loadings",
                    ),
                    use_container_width=True,
                )
            st.caption(
                "⚠️ Rolling PCA uses the current top-10 holdings throughout. "
                "Constituent composition changes over time are not reflected — "
                "interpret with caution over longer periods."
            )
