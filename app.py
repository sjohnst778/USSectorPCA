import streamlit as st
from datetime import date, timedelta

import pandas as pd

from pca.data import fetch_adjusted_close
from pca.returns import compute_returns, compute_relative_returns
from pca.analysis import run_pca, PCAResult
from pca.holdings import fetch_etf_holdings
from pca import plots

st.set_page_config(page_title="PCA Equity Analysis", layout="wide")
st.title("Equity Return PCA — S&P 500 Sectors")

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
BENCHMARK = "SPY"

SECTOR_NAMES = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Information Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}

# XLC from 2018, XLRE from 2015 — flag if date range extends before these
XLC_START = date(2018, 6, 19)
XLRE_START = date(2015, 10, 7)

# --- Sidebar controls ---
with st.sidebar:
    st.header("Configuration")

    preset_col, info_col = st.columns([5, 1])
    use_preset = preset_col.checkbox("Use S&P 500 sector preset", value=True)
    with info_col.popover("ⓘ"):
        st.markdown("**Sector ETF reference**")
        for ticker, name in SECTOR_NAMES.items():
            st.markdown(f"**{ticker}** — {name}")
        st.markdown(f"**{BENCHMARK}** — S&P 500 (benchmark)")

    if use_preset:
        ticker_input = ", ".join(SECTOR_ETFS)
        st.caption(f"Benchmark: {BENCHMARK} (SPY)")
        st.text_area("Tickers (read-only)", value=ticker_input, disabled=True, height=80)
        tickers = SECTOR_ETFS[:]
    else:
        ticker_input = st.text_area(
            "Tickers (comma-separated)",
            value=", ".join(SECTOR_ETFS),
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
def _pca_tabs(label: str, result: PCAResult, n_comp: int, returns: "pd.DataFrame") -> None:
    tab1, tab2, tab3, tab4 = st.tabs(["Scree", "Loadings", "Biplot", "Correlation"])
    with tab1:
        st.plotly_chart(plots.scree_plot(result), use_container_width=True)
    with tab2:
        st.plotly_chart(plots.loadings_heatmap(result), use_container_width=True)
        with st.expander("Loadings table"):
            st.dataframe(result.loadings.style.background_gradient(cmap="RdBu_r", axis=None))
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
        if start_date < XLC_START:
            st.warning(
                f"XLC (Communication Services) only has data from {XLC_START}. "
                "It will be included from that date; earlier rows will be dropped."
            )
        if start_date < XLRE_START:
            st.warning(
                f"XLRE (Real Estate) only has data from {XLRE_START}. "
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
    benchmark_series = all_returns[BENCHMARK]
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
        result_abs=result_abs,
        result_rel=result_rel,
        sector_tickers=sector_tickers,
        n_comp=n_comp,
        pca_kwargs=pca_kwargs,
        start_date=start_date,
        end_date=end_date,
        return_type=return_type,
    )
    st.session_state.pop("drill_results", None)   # clear stale drill-down on fresh run

if "pca" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run PCA**.")
    st.stop()

# --- Unpack persisted state ---
_s = st.session_state["pca"]
sector_returns   = _s["sector_returns"]
relative_returns = _s["relative_returns"]
result_abs       = _s["result_abs"]
result_rel       = _s["result_rel"]
sector_tickers   = _s["sector_tickers"]
n_comp           = _s["n_comp"]
pca_kwargs       = _s["pca_kwargs"]
start_date       = _s["start_date"]
end_date         = _s["end_date"]
return_type      = _s["return_type"]

# --- Summary ---
st.subheader("Data Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Sectors used", len(sector_tickers))
m2.metric("Trading days", sector_returns.shape[0])
m3.metric("Date range", f"{sector_returns.index[0].date()} → {sector_returns.index[-1].date()}")

st.divider()

# --- Dual PCA columns ---
col_abs, col_rel = st.columns(2)

with col_abs:
    st.subheader("Absolute Returns PCA")
    st.caption("PCA on raw sector returns — PC1 typically captures the broad market factor.")
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
    st.caption(f"PCA on sector returns minus {BENCHMARK} — isolates rotation dynamics beyond the market.")
    n_met = 4 if result_rel.shrinkage_coefficient is not None else 3
    r_cols = st.columns(n_met)
    r_cols[0].metric("Variance explained (all PCs)", f"{result_rel.cumulative_variance[-1]*100:.1f}%")
    r_cols[1].metric("PC1 explains", f"{result_rel.explained_variance_ratio[0]*100:.1f}%")
    r_cols[2].metric("Matrix", result_rel.matrix_type.capitalize())
    if result_rel.shrinkage_coefficient is not None:
        r_cols[3].metric("LW shrinkage α", f"{result_rel.shrinkage_coefficient:.3f}")
    _pca_tabs("rel", result_rel, n_comp, relative_returns)

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
        "relative to SPY — useful for seeing how stocks from different sectors interleave "
        "in factor space. When unchecked, a separate PCA is run for each ETF relative to "
        "that ETF's own return."
    ),
)

if merge_constituents:
    st.caption(
        "Merged mode: single PCA across all selected ETF constituents, relative to SPY. "
        "Sector column identifies the source ETF for each stock."
    )
else:
    st.caption(
        "Separate mode: one PCA per ETF, each relative to that ETF's own return. "
        "Holdings sourced from yfinance (top 10 by weight)."
    )

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

        available = [t for t in all_syms if t in c_prices.columns]
        if BENCHMARK not in c_prices.columns:
            st.warning("Could not fetch SPY for benchmarking.")
            st.session_state.pop("drill_results", None)
        elif len(available) < 3:
            st.warning("Fewer than 3 stocks have data across the selected ETFs.")
            st.session_state.pop("drill_results", None)
        else:
            c_all_rets = compute_returns(c_prices[available + [BENCHMARK]], method=return_type)
            c_rel_rets = compute_relative_returns(c_all_rets[available], c_all_rets[BENCHMARK])

            n_c_comp = min(2, len(available))
            c_result = run_pca(c_rel_rets, **{**pca_kwargs, "n_components": n_c_comp})

            loadings_df = _build_loadings_table(
                c_result,
                combined[["symbol", "name", "sector", "weight"]],
                n_c_comp,
            )
            # Per-ETF weight coverage for merged display
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
                }
            }
    else:
        drill_results = {}
        for etf, holdings in all_holdings.items():
            constituent_tickers = holdings["symbol"].tolist()
            fetch_list = list(dict.fromkeys(constituent_tickers + [etf]))

            with st.spinner(f"Fetching price data for {etf} constituents…"):
                c_prices = fetch_adjusted_close(fetch_list, str(start_date), str(end_date))

            available_constituents = [t for t in constituent_tickers if t in c_prices.columns]
            if etf not in c_prices.columns:
                drill_results[etf] = {"error": f"Could not retrieve {etf} price data for benchmarking."}
                continue
            if len(available_constituents) < 3:
                drill_results[etf] = {"error": f"Fewer than 3 constituents have data for {etf}."}
                continue

            c_all_rets = compute_returns(
                c_prices[available_constituents + [etf]], method=return_type
            )
            c_rel_rets = compute_relative_returns(c_all_rets[available_constituents], c_all_rets[etf])

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
            _render_loadings_table(data["loadings_df"], data["result"], data["n_c_comp"], extra_cols=["sector"])
        else:
            st.markdown(f"#### {key} — {SECTOR_NAMES.get(key, key)}")
            total_w = coverage.get(key, 0.0)
            st.caption(f"Top 10 holdings represent **{total_w:.1f}%** of {key}")
            _render_loadings_table(data["loadings_df"], data["result"], data["n_c_comp"])

        st.plotly_chart(plots.pc_scores_chart(data["result"]), use_container_width=True)
