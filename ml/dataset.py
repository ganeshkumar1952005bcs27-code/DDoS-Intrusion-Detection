"""
ml/dataset.py
-------------
Loads the IoTID20 dataset used in the paper (Section IV-A).

Real dataset (not auto-downloadable from this sandbox due to network
restrictions):
    https://sites.google.com/view/iot-network-intrusion-dataset/home

USAGE WITH THE REAL DATASET
----------------------------
Download "IoT Network Intrusion Dataset.csv", save it at the path set
in config.json -> dataset.csv_path (default: data/IoTID20.csv). Nothing
else needs to change -- load_dataset() will detect and use it.

FALLBACK (no internet / file not supplied)
-------------------------------------------
Transparently falls back to a SYNTHETIC dataset that mimics IoTID20's
structure (80 numeric network-flow features, benign vs. DDoS-family
attack traffic, class imbalance, correlated/redundant features) so the
full pipeline (PCA reduction -> train -> export) can be run and
demonstrated end-to-end without the original file.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

N_ORIGINAL_FEATURES = 80  # as stated in the paper ("original 80 features")
RANDOM_STATE = 42


def _synthetic_iotid20(n_samples: int = 20000,
                        n_features: int = N_ORIGINAL_FEATURES,
                        random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic network-flow style stand-in for IoTID20."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=40,
        n_repeated=8,
        n_clusters_per_class=3,
        weights=[0.62, 0.38],
        flip_y=0.01,
        class_sep=1.6,
        random_state=random_state,
    )
    cols = [f"feat_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    rng = np.random.default_rng(random_state)
    scale = rng.uniform(1, 500, size=n_features)
    df = df * scale
    df["Label"] = np.where(y == 1, "Anomaly", "Normal")
    return df


def load_dataset(csv_path: str = None, verbose: bool = True):
    """
    Returns (X: pd.DataFrame[float], y: pd.Series[int], feature_names, source)
    y = 1 for attack/anomaly traffic, 0 for benign/normal traffic.

    csv_path: explicit override. If None, reads config.json/
    config.example.json -> dataset.csv_path (resolved relative to the
    project root).
    """
    if csv_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(project_root, "config.json")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(project_root, "config.example.json")
        rel = "data/IoTID20.csv"
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            rel = cfg.get("dataset", {}).get("csv_path", rel)
        csv_path = rel if os.path.isabs(rel) else os.path.join(project_root, rel)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        source = f"real IoTID20 CSV ({csv_path})"
    else:
        df = _synthetic_iotid20()
        source = f"SYNTHETIC stand-in ({csv_path} not found)"

    label_col = None
    for candidate in ["Label", "label", "Cat", "class", "Class"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Could not find a label column in the dataset.")

    y_raw = df[label_col].astype(str).str.strip()
    y = (~y_raw.str.contains("normal", case=False)).astype(int)

    drop_cols = [c for c in df.columns if c in
                 ["Label", "label", "Cat", "Sub_Cat", "class", "Class",
                  "Flow_ID", "Src_IP", "Dst_IP", "Timestamp"]]
    X = df.drop(columns=drop_cols, errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")

    # IoTID20 (like other CICFlowMeter-derived datasets) contains genuine
    # Infinity values in rate columns such as Flow_Byts/s and Flow_Pkts/s,
    # produced whenever Flow_Duration is 0 (division by zero during flow
    # export). Left as inf, these corrupt every downstream statistic that
    # touches them (mean/std/median all become inf or nan, and later
    # comparisons involving them raise "invalid value encountered in
    # subtract" warnings). Convert inf -> NaN FIRST so the median/std
    # calculations below are computed only from finite values.
    n_inf = np.isinf(X.to_numpy(dtype="float64", na_value=np.nan)).sum()
    if n_inf > 0 and verbose:
        print(f"[dataset] Found {n_inf} Infinity values (e.g. rate columns "
              f"with Flow_Duration=0) -> converting to NaN before imputing.")
    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    std = X.std(numeric_only=True)
    zero_var_cols = std[std <= 1e-8].index.tolist()
    if zero_var_cols and verbose:
        print(f"[dataset] Dropping {len(zero_var_cols)} zero-variance "
              f"column(s): {zero_var_cols}")
    X = X.drop(columns=zero_var_cols)

    if verbose:
        print(f"[dataset] Source: {source}")
        print(f"[dataset] Shape: X={X.shape}, classes={dict(y.value_counts())}")

    return X.reset_index(drop=True), y.reset_index(drop=True), list(X.columns), source


if __name__ == "__main__":
    X, y, cols, source = load_dataset()
    print(X.head())
