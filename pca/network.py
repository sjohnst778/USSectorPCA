import pandas as pd
import networkx as nx


def build_mst(returns: pd.DataFrame) -> tuple[nx.Graph, pd.DataFrame]:
    """Build Minimum Spanning Tree from the pairwise return correlation matrix.

    Uses Mantegna distance d = 1 − ρ as the edge weight, so the MST connects
    the most-correlated pairs first.  The full correlation DataFrame is also
    returned because the plot function needs it for layout and hover text.
    """
    corr = returns.corr()
    tickers = corr.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = float(corr.iloc[i, j])
            G.add_edge(tickers[i], tickers[j], weight=1.0 - c, corr=c)
    mst = nx.minimum_spanning_tree(G, algorithm="kruskal", weight="weight")
    return mst, corr


def detect_communities(corr: pd.DataFrame) -> dict[str, int]:
    """Louvain community detection on the positive-correlation graph.

    Builds a graph where edge weights equal the raw correlation (only positive
    pairs get an edge), then runs the Louvain algorithm built into networkx.
    Returns a dict mapping ticker → integer community id (0-indexed).
    Falls back to one community per node if there are no positive-correlation edges.
    """
    tickers = corr.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = float(corr.iloc[i, j])
            if c > 0:
                G.add_edge(tickers[i], tickers[j], weight=c)

    if G.number_of_edges() == 0:
        return {t: i for i, t in enumerate(tickers)}

    communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    return {node: cid for cid, nodes in enumerate(communities) for node in nodes}
