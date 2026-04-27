import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import PCAResult


def correlation_heatmap(returns: "pd.DataFrame", title: str = "Correlation Matrix") -> go.Figure:
    import pandas as pd
    corr = returns.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
        text_auto=".2f",
        labels=dict(color="Correlation"),
    )
    fig.update_traces(textfont_size=10)
    fig.update_layout(
        coloraxis_colorbar=dict(thickness=12, len=0.8),
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def scree_plot(result: PCAResult) -> go.Figure:
    labels = [f"PC{i+1}" for i in range(result.components)]
    fig = go.Figure()
    fig.add_bar(x=labels, y=result.explained_variance_ratio * 100, name="Individual")
    fig.add_scatter(
        x=labels, y=result.cumulative_variance * 100,
        mode="lines+markers", name="Cumulative", yaxis="y"
    )
    fig.update_layout(
        title="Scree Plot",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def loadings_heatmap(result: PCAResult) -> go.Figure:
    fig = px.imshow(
        result.loadings,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
        title="PCA Loadings Heatmap",
        labels=dict(x="Component", y="Ticker", color="Loading"),
    )
    return fig


def biplot(result: PCAResult, pc_x: int = 1, pc_y: int = 2) -> go.Figure:
    """Scatter of scores for two PCs with loading arrows overlaid."""
    x_label, y_label = f"PC{pc_x}", f"PC{pc_y}"
    scores = result.scores
    loadings = result.loadings

    fig = go.Figure()
    fig.add_scatter(
        x=scores[x_label], y=scores[y_label],
        mode="markers", marker=dict(size=3, opacity=0.5, color="steelblue"),
        name="Observations",
    )

    scale = (scores[x_label].abs().max() + scores[y_label].abs().max()) / 2
    for ticker in loadings.index:
        lx, ly = loadings.loc[ticker, x_label] * scale, loadings.loc[ticker, y_label] * scale
        fig.add_annotation(
            x=lx, y=ly, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowcolor="crimson", arrowwidth=1.5,
        )
        fig.add_scatter(
            x=[lx], y=[ly], mode="text", text=[ticker],
            textposition="top center", showlegend=False,
            textfont=dict(size=10, color="crimson"),
        )

    fig.update_layout(
        title=f"Biplot ({x_label} vs {y_label})",
        xaxis_title=x_label, yaxis_title=y_label,
        showlegend=False,
    )
    return fig


def pc_scores_chart(result: PCAResult) -> go.Figure:
    """Cumulative PC scores over time — one subplot per component.

    Each subplot shows the cumulative sum of daily factor scores, giving a
    'factor index' that reveals how each latent factor has trended over the period.
    A zero reference line is drawn on each panel.
    """
    n = result.components
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"PC{i+1}  ({result.explained_variance_ratio[i]*100:.1f}% var)"
            for i in range(n)
        ],
        vertical_spacing=0.04,
    )

    for i in range(n):
        label = f"PC{i+1}"
        cumulative = result.scores[label].cumsum()
        color = colors[i % len(colors)]

        # Zero reference
        fig.add_hline(y=0, line_width=0.8, line_color="grey", line_dash="dot", row=i + 1, col=1)

        fig.add_scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            line=dict(color=color, width=1.5),
            name=label,
            showlegend=False,
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title="Cumulative PC Score Projections",
        height=220 * n,
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(showticklabels=True, row=n, col=1)

    return fig


def rolling_variance_chart(ev_df: "pd.DataFrame", title: str = "Rolling Explained Variance") -> go.Figure:
    """Line chart of rolling explained variance ratio for each PC over time."""
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, col in enumerate(ev_df.columns):
        fig.add_scatter(
            x=ev_df.index,
            y=ev_df[col] * 100,
            mode="lines",
            name=col,
            line=dict(color=colors[i % len(colors)], width=1.5),
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Variance Explained (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=320,
        margin=dict(t=60, b=40),
    )
    return fig


def mst_plot(
    mst: "nx.Graph",
    corr: "pd.DataFrame",
    communities: "dict[str, int]",
    names: "dict[str, str]",
    title: str = "Minimum Spanning Tree",
) -> go.Figure:
    """MST network graph with nodes coloured by Louvain community.

    Layout is computed from the full pairwise distance matrix (not just MST edges)
    using Kamada-Kawai, so geometrically close nodes are highly correlated even
    if not directly connected in the tree.  Edge thickness scales with |correlation|.
    """
    import networkx as nx

    tickers = list(corr.columns)
    G_full = nx.Graph()
    G_full.add_nodes_from(tickers)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            G_full.add_edge(tickers[i], tickers[j], weight=1.0 - float(corr.iloc[i, j]))
    pos = nx.kamada_kawai_layout(G_full, weight="weight")

    n_comm = max(communities.values(), default=0) + 1
    palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.Dark24
    )
    comm_colors = {i: palette[i % len(palette)] for i in range(n_comm)}

    fig = go.Figure()

    for u, v, data in mst.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        c = data["corr"]
        fig.add_scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=0.5 + 4.0 * abs(c), color="rgba(140,140,140,0.65)"),
            hoverinfo="none",
            showlegend=False,
        )
        fig.add_scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            mode="markers",
            marker=dict(size=14, opacity=0),
            hovertemplate=f"<b>{u} – {v}</b><br>Correlation: {c:+.3f}<extra></extra>",
            showlegend=False,
        )

    comm_groups: dict[int, list[str]] = {}
    for node in mst.nodes():
        comm_groups.setdefault(communities.get(node, 0), []).append(node)

    for cid, nodes in sorted(comm_groups.items()):
        fig.add_scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            text=nodes,
            textposition="top center",
            textfont=dict(size=10, color="#333333"),
            hovertemplate=[
                f"<b>{n}</b><br>{names.get(n, '')}<br>Community {cid + 1}<extra></extra>"
                for n in nodes
            ],
            marker=dict(size=20, color=comm_colors[cid], line=dict(width=2, color="white")),
            name=f"Community {cid + 1}",
            legendgroup=f"comm_{cid}",
        )

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        plot_bgcolor="white",
        height=460,
        margin=dict(t=80, b=20, l=20, r=20),
    )
    return fig


def rolling_loadings_heatmap(loadings_df: "pd.DataFrame", title: str = "Rolling PC1 Loadings") -> go.Figure:
    """Heatmap of rolling loadings: x=date, y=ticker/name, colour=loading value."""
    fig = px.imshow(
        loadings_df.T,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
        labels=dict(x="Date", y="", color="Loading"),
    )
    fig.update_layout(
        height=320,
        margin=dict(t=60, b=40),
        coloraxis_colorbar=dict(thickness=12, len=0.8),
    )
    return fig
