"""
ml/train_export.py
-------------------
Trains and exports the FINAL deployed model:

    raw features (O) -> StandardScaler -> PCA(Fm) -> DecisionTreeClassifier

Why this pipeline (and not a tree on raw features)?
  - HANP (ml/hanp.py, Algorithm 1) ranks Decision Tree #1 for TinyML
    suitability on low-power SD-IoT devices.
  - The paper trains that tree on the PCA-reduced feature set
    (Algorithm 2 / Section IV-B-C), so the deployed model here does the
    same, instead of silently switching to raw top-K features.
  - The scaler + PCA transform is just a matrix-vector multiply
    (O x Fm floats), which is trivial for an ESP32 (240 MHz, 320 KB
    RAM) -- so we export the FULL pipeline to C, not an approximation.

This produces TWO equivalent representations of the exact same model,
for the two ways this project can be run (see README.md "Hardware vs.
software" section):

  1. outputs/scaler.joblib, outputs/pca.joblib, outputs/decision_tree.joblib
     -> loaded by scripts/simulate_edge_node.py (the SOFTWARE / no-ESP32
        path -- runs on your PC with scikit-learn).

  2. esp32/generated_model.h
     -> flashed onto a REAL ESP32 (the HARDWARE path) -- pure C, no
        scikit-learn/numpy needed on-device.

Both paths compute IDENTICAL predictions given identical raw feature
input, because they encode the same fitted scaler+pca+tree.

Also writes outputs/model_metadata.json (feature order, Fm, metrics --
read by both deployment paths so they agree on input format).
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from ml.dataset import load_dataset
from ml.pca_reduction import find_optimal_fm, transform_with_fm

RANDOM_STATE = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
ESP32_DIR = os.path.join(PROJECT_ROOT, "esp32")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ESP32_DIR, exist_ok=True)


def _load_model_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, "config.example.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    return cfg


def train_pipeline(X: pd.DataFrame, y: pd.Series, cfg: dict):
    tolerance = cfg.get("pca", {}).get("variance_tolerance", 0.01)
    model_cfg = cfg.get("model", {})
    # IMPORTANT: these two must match None/"model" key ABSENCE, not a
    # baked-in fallback number, so this exported model reproduces the
    # SAME DecisionTreeClassifier settings as the "Reduced (PCA)" row
    # benchmarked in ml/train_evaluate.py (unrestricted depth, no class
    # weighting) -- that is what gets you paper-comparable metrics.
    # Set config.json -> model.max_tree_depth to an integer ONLY if you
    # specifically need a smaller/faster tree for flash-constrained
    # hardware; expect measurably lower recall/accuracy if you do.
    max_depth = model_cfg.get("max_tree_depth", None)
    class_weight = model_cfg.get("class_weight", None)
    test_size = model_cfg.get("test_size", 0.25)
    random_state = model_cfg.get("random_state", RANDOM_STATE)

    Fm, cum_var, scaler, _ = find_optimal_fm(X.values, tolerance=tolerance)
    X_reduced, pca, scaler = transform_with_fm(X.values, Fm, scaler)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y.values, test_size=test_size, stratify=y.values,
        random_state=random_state)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state,
                                  class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "Precision(%)": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall(%)": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Accuracy(%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "F1(%)": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "TreeDepth": int(clf.get_depth()),
        "TreeLeaves": int(clf.get_n_leaves()),
        "Fm": int(Fm),
        "OriginalFeatures": int(X.shape[1]),
    }
    return clf, scaler, pca, Fm, metrics


def save_joblib_artifacts(clf, scaler, pca):
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(pca, os.path.join(OUT_DIR, "pca.joblib"))
    joblib.dump(clf, os.path.join(OUT_DIR, "decision_tree.joblib"))
    print(f"[export] Saved scaler.joblib, pca.joblib, decision_tree.joblib -> {OUT_DIR}")


def save_metadata(feature_names, Fm, metrics, path):
    metadata = {
        "raw_feature_names": feature_names,
        "num_raw_features": len(feature_names),
        "num_pca_components": Fm,
        "metrics": metrics,
        "pipeline": "raw -> StandardScaler -> PCA(Fm) -> DecisionTreeClassifier",
        "note": "Both esp32/generated_model.h and scripts/simulate_edge_node.py "
                "expect raw feature vectors of length num_raw_features, IN THIS "
                "EXACT ORDER (raw_feature_names).",
    }
    with open(path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"[export] Saved metadata -> {path}")


def tree_to_c_if_else(clf: DecisionTreeClassifier, function_name="predict_from_pca") -> str:
    """Emit the tree as nested if/else, operating on the Fm-length PCA
    component array (float pc[NUM_COMPONENTS])."""
    tree = clf.tree_
    lines = []

    def recurse(node, depth):
        indent = "    " * (depth + 1)
        if tree.feature[node] != -2:
            idx = tree.feature[node]
            threshold = tree.threshold[node]
            lines.append(f"{indent}if (pc[{idx}] <= {threshold:.8f}f) {{")
            recurse(tree.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(tree.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            counts = tree.value[node][0]
            predicted = int(np.argmax(counts))
            lines.append(f"{indent}return {predicted};  // samples: {counts.tolist()}")

    lines.append(f"int {function_name}(const float *pc) {{")
    recurse(0, 0)
    lines.append("}")
    return "\n".join(lines)


def c_float_array(name: str, values: np.ndarray) -> str:
    body = ", ".join(f"{v:.8f}f" for v in values.ravel())
    return f"static const float {name}[{values.size}] = {{{body}}};"


def c_float_matrix(name: str, matrix: np.ndarray) -> str:
    """matrix shape (Fm, O) -> flattened row-major C array + a helper
    to index it as MATRIX[row * O + col]."""
    rows, cols = matrix.shape
    body = ", ".join(f"{v:.8f}f" for v in matrix.ravel())
    return f"static const float {name}[{rows * cols}] = {{{body}}}; // shape ({rows}, {cols}), row-major"


def write_generated_model_h(scaler, pca, clf, feature_names, Fm, metrics):
    O = len(feature_names)
    feature_list_c = ", ".join(f'"{f}"' for f in feature_names)

    scaler_mean = c_float_array("SCALER_MEAN", scaler.mean_)
    scaler_scale = c_float_array("SCALER_SCALE", scaler.scale_)
    pca_mean = c_float_array("PCA_MEAN", pca.mean_)
    pca_components = c_float_matrix("PCA_COMPONENTS", pca.components_)  # (Fm, O)
    tree_code = tree_to_c_if_else(clf)

    header = f"""// ============================================================
// generated_model.h -- AUTO-GENERATED, do not edit by hand.
// Regenerate with: python3 -m ml.train_export   (from project root)
//
// Deployed pipeline: raw[{O}] -> StandardScaler -> PCA[{Fm}] -> DecisionTree
// (HANP Algorithm-1 picks Decision Tree as the top TinyML choice; PCA
//  reduction follows Algorithm 2 -- see ml/hanp.py and ml/pca_reduction.py)
//
// Held-out test set performance:
//   Precision={metrics['Precision(%)']}%  Recall={metrics['Recall(%)']}%
//   Accuracy={metrics['Accuracy(%)']}%    F1={metrics['F1(%)']}%
//   Tree depth={metrics['TreeDepth']}  Leaves={metrics['TreeLeaves']}
// ============================================================
#ifndef GENERATED_MODEL_H
#define GENERATED_MODEL_H

#define NUM_RAW_FEATURES {O}
#define NUM_PCA_COMPONENTS {Fm}

// Human-readable names of the {O} raw features expected in the
// `raw[]` array, IN THIS EXACT ORDER. You must compute/extract these
// same flow statistics on-device (or receive them from a companion
// flow-metering process) before calling classify_flow().
static const char* RAW_FEATURE_NAMES[NUM_RAW_FEATURES] = {{{feature_list_c}}};

// ---- StandardScaler ----
{scaler_mean}
{scaler_scale}

// ---- PCA ----
{pca_mean}
{pca_components}

// scaled[i]     = (raw[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]
// centered[i]   = scaled[i] - PCA_MEAN[i]
// pc[j]         = sum_i centered[i] * PCA_COMPONENTS[j*NUM_RAW_FEATURES + i]
static inline void transform_to_pca(const float *raw, float *pc) {{
    float centered[NUM_RAW_FEATURES];
    for (int i = 0; i < NUM_RAW_FEATURES; i++) {{
        float scaled = (raw[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
        centered[i] = scaled - PCA_MEAN[i];
    }}
    for (int j = 0; j < NUM_PCA_COMPONENTS; j++) {{
        float acc = 0.0f;
        for (int i = 0; i < NUM_RAW_FEATURES; i++) {{
            acc += centered[i] * PCA_COMPONENTS[j * NUM_RAW_FEATURES + i];
        }}
        pc[j] = acc;
    }}
}}

// ---- Decision Tree (operates on the PCA component array) ----
// Returns 1 = ATTACK (DDoS/DoS/Mirai/ARP-spoof family), 0 = BENIGN.
{tree_code}

// Convenience wrapper: raw features in, class out.
static inline int classify_flow(const float *raw) {{
    float pc[NUM_PCA_COMPONENTS];
    transform_to_pca(raw, pc);
    return predict_from_pca(pc);
}}

#endif // GENERATED_MODEL_H
"""
    path = os.path.join(ESP32_DIR, "generated_model.h")
    with open(path, "w") as fh:
        fh.write(header)
    print(f"[export] Wrote {path}")
    return path


def main():
    cfg = _load_model_config()
    X, y, feature_names, source = load_dataset()

    clf, scaler, pca, Fm, metrics = train_pipeline(X, y, cfg)
    print(f"[export] Final pipeline metrics: {metrics}")

    save_joblib_artifacts(clf, scaler, pca)
    save_metadata(feature_names, Fm, metrics, os.path.join(OUT_DIR, "model_metadata.json"))
    write_generated_model_h(scaler, pca, clf, feature_names, Fm, metrics)

    print("[export] Done. Hardware path -> flash esp32/esp32_tinyml_ids.ino "
          "(includes generated_model.h). Software path -> "
          "scripts/simulate_edge_node.py loads the .joblib files directly.")


if __name__ == "__main__":
    main()
