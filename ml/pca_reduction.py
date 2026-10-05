"""
ml/pca_reduction.py
--------------------
Implements Algorithm 2 of the paper: find the smallest number of
principal components Fm (Fm < O, O = original feature count) such that
the average squared projection error, normalized by total variance,
stays below a tolerance (default 0.01, i.e. >= 99% variance retained):

    (1/n) * sum ||O_i - O_i_approx||^2
    ----------------------------------   <= tolerance
    (1/n) * sum ||O_i||^2

Equivalently (and what we compute numerically): keep increasing the
number of principal components until cumulative explained variance
ratio >= 1 - tolerance.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def find_optimal_fm(X: np.ndarray, tolerance: float = 0.01, max_components: int = None):
    """
    Algorithm 2: scan Fm = 1..O and return the smallest Fm for which the
    normalized squared projection error <= tolerance.
    Returns: (Fm, cumulative_explained_variance_ratio_array, scaler, full_pca)
    """
    O = X.shape[1]
    max_components = max_components or O

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA(n_components=min(max_components, X.shape[0], O), random_state=0)
    pca_full.fit(X_scaled)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    Fm = int(np.searchsorted(cum_var, 1 - tolerance) + 1)
    Fm = max(1, min(Fm, len(cum_var)))

    print(f"[PCA/Algorithm 2] Original features O = {O}")
    print(f"[PCA/Algorithm 2] Reduced features Fm = {Fm} "
          f"(retains {cum_var[Fm-1]*100:.2f}% variance, tolerance={tolerance})")
    print(f"[PCA/Algorithm 2] Reduction ratio R = Fm/O = {Fm/O:.3f}  (Eq. 11)")

    return Fm, cum_var, scaler, pca_full


def transform_with_fm(X: np.ndarray, Fm: int, scaler: StandardScaler = None):
    """Fit (or reuse) a scaler + refit PCA(Fm), return the reduced matrix."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    pca = PCA(n_components=Fm, random_state=0)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca, scaler


def save_variance_csv(cum_var: np.ndarray, Fm: int, out_path: str):
    df = pd.DataFrame({
        "n_components": np.arange(1, len(cum_var) + 1),
        "cumulative_explained_variance": cum_var,
        "is_selected_Fm": [i == Fm for i in range(1, len(cum_var) + 1)],
    })
    df.to_csv(out_path, index=False)
    print(f"[PCA/Algorithm 2] Saved variance curve to {out_path}")


if __name__ == "__main__":
    from ml.dataset import load_dataset

    X, y, cols, source = load_dataset()
    Fm, cum_var, scaler, pca_full = find_optimal_fm(X.values, tolerance=0.01)
    X_reduced, pca, _ = transform_with_fm(X.values, Fm, scaler)
    print("Reduced feature matrix shape:", X_reduced.shape)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_variance_csv(cum_var, Fm, os.path.join(out_dir, "pca_variance.csv"))
