from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class PCAResult:
    components: int
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame          # shape: (tickers, components)
    scores: pd.DataFrame            # shape: (dates, components)
    cumulative_variance: np.ndarray
    shrinkage_coefficient: float | None = field(default=None)
    matrix_type: str = field(default="correlation")   # 'correlation' or 'covariance'
    standardized: bool = field(default=True)


def run_pca(
    returns: pd.DataFrame,
    n_components: int,
    use_shrinkage: bool = False,
    standardize: bool = True,
    matrix_type: str = "correlation",
) -> PCAResult:
    """Fit PCA on the correlation or covariance matrix of sector returns.

    Steps:
      1. Demean (always).
      2. Optionally scale to unit variance (standardize).
      3. Compute the chosen matrix type, optionally with Ledoit-Wolf shrinkage.
      4. Eigendecompose to get loadings and explained variance.
      5. Project demeaned (scaled) data onto loadings to get scores.

    Note: when standardize=True the covariance of the scaled data equals the
    correlation matrix, so the matrix_type toggle has no effect in that case.
    """
    X = returns.values.astype(float)
    X = X - X.mean(axis=0)  # demean always

    if standardize:
        stds = X.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0   # guard against zero-variance columns
        X = X / stds
        effective_matrix = "correlation"  # standardised cov = correlation
    else:
        effective_matrix = matrix_type

    shrinkage_coef: float | None = None

    if use_shrinkage:
        lw = LedoitWolf()
        lw.fit(X)
        cov = lw.covariance_
        shrinkage_coef = float(lw.shrinkage_)
    else:
        cov = np.cov(X.T)

    if effective_matrix == "correlation":
        # Convert to correlation regardless of how cov was estimated
        d = np.sqrt(np.diag(cov))
        d[d == 0] = 1.0
        matrix = cov / np.outer(d, d)
        # Clip to [-1, 1] to guard against floating-point noise
        matrix = np.clip(matrix, -1.0, 1.0)
    else:
        matrix = cov

    # Eigendecomposition — eigh for symmetric matrices, returns ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.clip(eigenvalues, 0, None)

    total = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total if total > 0 else eigenvalues

    component_labels = [f"PC{i+1}" for i in range(n_components)]
    loadings_arr = eigenvectors[:, :n_components]   # (N, k)
    scores_arr = X @ loadings_arr                   # (T, k)

    loadings = pd.DataFrame(loadings_arr, index=returns.columns, columns=component_labels)
    scores = pd.DataFrame(scores_arr, index=returns.index, columns=component_labels)

    return PCAResult(
        components=n_components,
        explained_variance_ratio=explained_variance_ratio[:n_components],
        loadings=loadings,
        scores=scores,
        cumulative_variance=np.cumsum(explained_variance_ratio[:n_components]),
        shrinkage_coefficient=shrinkage_coef,
        matrix_type=effective_matrix,
        standardized=standardize,
    )
