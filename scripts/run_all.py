"""
scripts/run_all.py
--------------------
Single entrypoint for the entire OFFLINE side of the project. Run this
FIRST, before touching the ESP32 or the controller.

Steps:
  1. HANP (ml/hanp.py, Algorithm 1)      -> outputs/hanp_ranking.csv
  2. Load dataset (ml/dataset.py)         -> real IoTID20.csv or synthetic
  3. Benchmark DT/RF/SVM/KNN
     (ml/train_evaluate.py)               -> outputs/ml_performance_results.csv
  4. Train + export final pipeline
     (ml/train_export.py)                 -> outputs/{scaler,pca,decision_tree}.joblib
                                              outputs/model_metadata.json
                                              outputs/pca_variance.csv
                                              esp32/generated_model.h

USAGE (from the project root)
  python3 scripts/run_all.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.hanp import run_hanp  # noqa: E402
from ml.dataset import load_dataset  # noqa: E402
from ml.pca_reduction import find_optimal_fm, transform_with_fm, save_variance_csv  # noqa: E402
from ml.train_evaluate import evaluate_models  # noqa: E402
import ml.train_export as train_export  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 1: HANP (AHP + ANP) ranking of ML algorithms for TinyML")
    print("=" * 70)
    hanp_ranking, crit_weights, supermatrix = run_hanp()
    hanp_ranking.to_csv(os.path.join(OUT_DIR, "hanp_ranking.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 2: Load dataset (IoTID20 or synthetic stand-in)")
    print("=" * 70)
    X, y, cols, source = load_dataset()

    print("\n" + "=" * 70)
    print("STEP 3: PCA reduction analysis (Algorithm 2)")
    print("=" * 70)
    Fm, cum_var, scaler, pca_full = find_optimal_fm(X.values, tolerance=0.01)
    X_reduced, _, _ = transform_with_fm(X.values, Fm, scaler)
    save_variance_csv(cum_var, Fm, os.path.join(OUT_DIR, "pca_variance.csv"))

    print("\n" + "=" * 70)
    print("STEP 4: Benchmark DT / RF / SVM / KNN (original vs PCA-reduced)")
    print("=" * 70)
    print("\n-- Original features --")
    results_orig = evaluate_models(X.values, y.values, "Original")
    print("\n-- PCA-reduced features --")
    results_reduced = evaluate_models(X_reduced, y.values, "Reduced (PCA)")

    import pandas as pd
    all_results = pd.concat([results_orig, results_reduced], ignore_index=True)
    all_results.to_csv(os.path.join(OUT_DIR, "ml_performance_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 5: Train & export the FINAL deployed pipeline")
    print("=" * 70)
    train_export.main()

    print("\n" + "=" * 70)
    print("Cross-check: HANP ranking vs. empirical F1 ranking")
    print("=" * 70)
    empirical_rank = (all_results[all_results.FeatureSet == "Original"]
                       .sort_values("F1(%)", ascending=False)["Algorithm"].tolist())
    print(f"HANP predicted ranking : {hanp_ranking['Algorithm'].tolist()}")
    print(f"Empirical F1 ranking   : {empirical_rank}")

    print("\n" + "=" * 70)
    print("DONE. All tables/figures saved to outputs/. Next steps:")
    print("  - HARDWARE: flash esp32/esp32_tinyml_ids.ino (+ generated_model.h)")
    print("  - SOFTWARE: run scripts/simulate_edge_node.py instead")
    print("  - Start controller/ryu_mitigation.py (ryu-manager)")
    print("  - (optional) test with network/mininet_topology.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
