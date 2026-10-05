import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"
output_folder = project_folder / "outputs"
model_folder = project_folder / "models"

output_folder.mkdir(exist_ok=True)
model_folder.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_train = pd.read_csv(data_folder / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(data_folder / "y_test.csv").iloc[:, 0]

print("=" * 70)
print("TINYML COMPARISON: 10 PCA vs 31 PCA")
print("=" * 70)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# --------------------------------------------------
# 3. StandardScaler
# --------------------------------------------------

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed.")

# --------------------------------------------------
# 4. Compare two PCA configurations
# --------------------------------------------------

pca_sizes = [10, 31]

results = []

for n_components in pca_sizes:

    print("\n" + "-" * 70)
    print(f"TESTING {n_components} PCA COMPONENTS")
    print("-" * 70)

    # --------------------------------------------------
    # PCA
    # --------------------------------------------------

    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    variance = (
        np.sum(pca.explained_variance_ratio_) * 100
    )

    print(
        f"Variance retained: {variance:.4f}%"
    )

    # --------------------------------------------------
    # Train Decision Tree
    # --------------------------------------------------

    print("Training Decision Tree...")

    start_train = time.perf_counter()

    dt = DecisionTreeClassifier(
        random_state=42
    )

    dt.fit(X_train_pca, y_train)

    train_time = (
        time.perf_counter() - start_train
    )

    # --------------------------------------------------
    # Prediction timing
    # --------------------------------------------------

    print("Measuring inference time...")

    start_predict = time.perf_counter()

    y_pred = dt.predict(X_test_pca)

    predict_time = (
        time.perf_counter() - start_predict
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------
    # Tree complexity
    # --------------------------------------------------

    tree_depth = dt.get_depth()
    node_count = dt.tree_.node_count
    leaf_count = dt.get_n_leaves()

    # --------------------------------------------------
    # Model size using joblib
    # --------------------------------------------------

    model_path = (
        model_folder /
        f"decision_tree_{n_components}pc.pkl"
    )

    joblib.dump(dt, model_path)

    model_size_kb = (
        model_path.stat().st_size / 1024
    )

    # --------------------------------------------------
    # Per-sample inference time
    # --------------------------------------------------

    samples = len(X_test_pca)

    inference_ms_total = predict_time * 1000

    inference_us_per_sample = (
        predict_time / samples * 1_000_000
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\nRESULTS")
    print(f"Accuracy            : {accuracy * 100:.4f}%")
    print(f"Precision           : {precision * 100:.4f}%")
    print(f"Recall              : {recall * 100:.4f}%")
    print(f"F1 Score            : {f1 * 100:.4f}%")

    print(f"\nTree depth          : {tree_depth}")
    print(f"Tree nodes          : {node_count}")
    print(f"Leaf nodes          : {leaf_count}")

    print(f"\nTraining time       : {train_time:.4f} seconds")
    print(
        f"Test inference time : "
        f"{inference_ms_total:.4f} ms"
    )
    print(
        f"Per-sample inference: "
        f"{inference_us_per_sample:.4f} µs"
    )

    print(
        f"Saved model size    : "
        f"{model_size_kb:.2f} KB"
    )

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append({
        "PCA_Components": n_components,
        "Variance_Retained_%": variance,
        "Accuracy_%": accuracy * 100,
        "Precision_%": precision * 100,
        "Recall_%": recall * 100,
        "F1_%": f1 * 100,
        "Tree_Depth": tree_depth,
        "Tree_Nodes": node_count,
        "Leaf_Nodes": leaf_count,
        "Model_Size_KB": model_size_kb,
        "Training_Time_s": train_time,
        "Inference_Time_ms": inference_ms_total,
        "Inference_us_per_sample":
            inference_us_per_sample
    })

# --------------------------------------------------
# 5. Final comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL 10 PC vs 31 PC COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# --------------------------------------------------
# 6. Save comparison
# --------------------------------------------------

comparison_path = (
    output_folder /
    "tinyml_10_vs_31_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nComparison saved to:")
print(comparison_path)

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)