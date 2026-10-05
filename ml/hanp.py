"""
hanp.py
--------
Implements the Hybrid Analytic Network Process (HANP) described in
"DDoS Intrusions Detection in Low Power SD-IoT Devices Leveraging
Effective Machine Learning" (Ali et al., IEEE TCE 2025), Algorithm 1.

HANP = AHP (for criteria weights) + ANP (for alternative/algorithm scores)

Criteria (Eq. 1):      RE, MC, TIS, ADP, RDDS, DRBD
Alternatives (Eq. 2):  DT, RF, SVM, KNN

Workflow:
  1. Build a pairwise comparison matrix for the criteria (Saaty 1-9 scale).
  2. Compute normalized eigenvector weights for the criteria (AHP), Eq. 5-6.
  3. Compute the consistency ratio (CR) to validate judgments, Eq. 7-10.
  4. For every criterion, build a pairwise comparison matrix between the
     four ML algorithms (ANP) and compute their local priority vector.
  5. Aggregate: overall_score(alt) = sum_c  weight(c) * local_priority(alt, c)
  6. Rank alternatives by overall_score (Algorithm 1, steps 11-17).
"""

import numpy as np
import pandas as pd

CRITERIA = ["RE", "MC", "TIS", "ADP", "RDDS", "DRBD"]
ALTERNATIVES = ["DT", "RF", "SVM", "KNN"]

# Saaty's Random Index (RI) table, indexed by matrix size n
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12,
                 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def _principal_eigenvector(matrix: np.ndarray) -> np.ndarray:
    """Return the normalized principal eigenvector (priority weights)."""
    eigvals, eigvecs = np.linalg.eig(matrix)
    idx = np.argmax(eigvals.real)
    vec = np.abs(eigvecs[:, idx].real)
    return vec / vec.sum()


def _consistency_ratio(matrix: np.ndarray, weights: np.ndarray) -> float:
    """Eq. (7)-(10): CI = (lambda_max - n) / (n - 1); CR = CI / RI."""
    n = matrix.shape[0]
    weighted_sum = matrix @ weights
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return ci, cr


def ahp_weights(pairwise_matrix: np.ndarray, labels=CRITERIA, verbose=True):
    """Step 1-3 of Algorithm 1: criteria weights via AHP."""
    weights = _principal_eigenvector(pairwise_matrix)
    ci, cr = _consistency_ratio(pairwise_matrix, weights)
    if verbose:
        print(f"[AHP] Criteria weights: "
              + ", ".join(f"{l}={w:.3f}" for l, w in zip(labels, weights)))
        print(f"[AHP] Consistency Index (CI) = {ci:.4f}, "
              f"Consistency Ratio (CR) = {cr:.4f} "
              f"({'ACCEPTABLE' if cr <= 0.1 else 'REVISE JUDGMENTS'})")
    return weights, ci, cr


def anp_local_priorities(pairwise_matrix: np.ndarray, labels=ALTERNATIVES):
    """Step 4-5 of Algorithm 1: local priority vector of alternatives
    for ONE criterion, using the ANP pairwise comparison matrix."""
    weights = _principal_eigenvector(pairwise_matrix)
    ci, cr = _consistency_ratio(pairwise_matrix, weights)
    return weights, ci, cr


def default_criteria_matrix():
    """
    Example pairwise comparison matrix for (RE, MC, TIS, ADP, RDDS, DRBD).
    Values follow Saaty's 1-9 scale (Table II of the paper). Edit these
    numbers to reflect your own expert judgment; the rest of the pipeline
    is fully driven by this matrix.
    Rationale used here: on a low-power IoT / TinyML device, Resource
    Efficiency (RE) and Model Complexity (MC) matter most, followed by
    inference speed (TIS) and dimensionality-reduction friendliness
    (DRBD); raw accuracy (ADP) and robustness to distribution shift
    (RDDS) are important but secondary given the "tiny" constraint.
    """
    n = len(CRITERIA)
    M = np.ones((n, n))
    # index: RE=0, MC=1, TIS=2, ADP=3, RDDS=4, DRBD=5
    pairs = {
        (0, 1): 2, (0, 2): 3, (0, 3): 2, (0, 4): 4, (0, 5): 3,
        (1, 2): 2, (1, 3): 1, (1, 4): 3, (1, 5): 2,
        (2, 3): 1 / 2, (2, 4): 2, (2, 5): 1,
        (3, 4): 2, (3, 5): 2,
        (4, 5): 1 / 2,
    }
    for (i, j), v in pairs.items():
        M[i, j] = v
        M[j, i] = 1 / v
    return M


def default_alternative_matrices():
    """
    Example ANP pairwise comparison matrices, one per criterion, between
    (DT, RF, SVM, KNN). Values are illustrative and should be replaced
    with the analyst's own judgments or with data-driven scores
    (e.g., normalized benchmark results) if available.
    """
    def build(pairs):
        n = len(ALTERNATIVES)
        M = np.ones((n, n))
        for (i, j), v in pairs.items():
            M[i, j] = v
            M[j, i] = 1 / v
        return M

    # index: DT=0, RF=1, SVM=2, KNN=3
    matrices = {
        # Resource Efficiency: DT lightest, KNN heaviest at inference
        "RE":   build({(0, 1): 2, (0, 2): 3, (0, 3): 4,
                        (1, 2): 2, (1, 3): 3, (2, 3): 2}),
        # Model Complexity: DT simplest
        "MC":   build({(0, 1): 2, (0, 2): 2, (0, 3): 3,
                        (1, 2): 1, (1, 3): 2, (2, 3): 2}),
        # Training/Inference Speed: DT & KNN fast to "train" (KNN is lazy)
        "TIS":  build({(0, 1): 2, (0, 2): 3, (0, 3): 1 / 1.5,
                        (1, 2): 2, (1, 3): 1 / 2, (2, 3): 1 / 3}),
        # Accuracy/Detection Performance: RF & DT strong ensembles/trees
        "ADP":  build({(0, 1): 1 / 1.2, (0, 2): 2, (0, 3): 1.5,
                        (1, 2): 2, (1, 3): 2, (2, 3): 1}),
        # Robustness to Data Distribution Shift: RF most robust (ensemble)
        "RDDS": build({(0, 1): 1 / 2, (0, 2): 1.5, (0, 3): 1.5,
                        (1, 2): 3, (1, 3): 3, (2, 3): 1}),
        # Dimensionality Reduction friendliness: DT assumed to work best
        # with reduced (PCA) feature sets (paper's stated assumption)
        "DRBD": build({(0, 1): 1.5, (0, 2): 2, (0, 3): 2,
                        (1, 2): 1.5, (1, 3): 1.5, (2, 3): 1}),
    }
    return matrices


def run_hanp(criteria_matrix=None, alternative_matrices=None, verbose=True):
    """
    Full Algorithm 1 pipeline.
    Returns a DataFrame with the overall HANP score & rank per algorithm.
    """
    if criteria_matrix is None:
        criteria_matrix = default_criteria_matrix()
    if alternative_matrices is None:
        alternative_matrices = default_alternative_matrices()

    crit_weights, c_ci, c_cr = ahp_weights(criteria_matrix, CRITERIA, verbose)

    local_priorities = {}
    for crit in CRITERIA:
        w, ci, cr = anp_local_priorities(alternative_matrices[crit], ALTERNATIVES)
        local_priorities[crit] = w
        if verbose:
            print(f"[ANP:{crit}] local priorities: "
                  + ", ".join(f"{a}={v:.3f}" for a, v in zip(ALTERNATIVES, w))
                  + f"  (CR={cr:.3f})")

    # Unweighted supermatrix: rows = alternatives, cols = criteria
    supermatrix = np.array([local_priorities[c] for c in CRITERIA]).T  # (alts x criteria)

    # Weighted supermatrix = supermatrix * criteria weights (column-wise)
    weighted = supermatrix * crit_weights  # broadcast over columns

    overall_scores = weighted.sum(axis=1)
    overall_scores = overall_scores / overall_scores.sum()  # normalize

    result = pd.DataFrame({
        "Algorithm": ALTERNATIVES,
        "HANP_Score": overall_scores
    }).sort_values("HANP_Score", ascending=False).reset_index(drop=True)
    result["Rank"] = result.index + 1

    if verbose:
        print("\n[HANP] Final ranking of ML algorithms for TinyML in SD-IoT:")
        print(result.to_string(index=False))

    return result, crit_weights, supermatrix


if __name__ == "__main__":
    ranking, weights, supermatrix = run_hanp()
