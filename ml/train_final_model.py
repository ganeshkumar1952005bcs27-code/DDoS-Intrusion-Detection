import pandas as pd
import numpy as np
import joblib
import json
import time

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data"
model_folder = project_folder / "models"
output_folder = project_folder / "outputs"

model_folder.mkdir(exist_ok=True)
output_folder.mkdir(exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL TINYML MODEL TRAINING")
print("=" * 70)

print("\nLoading training and testing data...")

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_train = pd.read_csv(
    data_folder / "y_train.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    data_folder / "y_test.csv"
).iloc[:, 0]

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# ============================================================
# 3. SAVE EXACT FEATURE ORDER
# ============================================================

feature_names = X_train.columns.tolist()

with open(
    model_folder / "feature_names.json",
    "w"
) as f:

    json.dump(
        feature_names,
        f,
        indent=4
    )

print("\nNumber of input features:", len(feature_names))

# ============================================================
# 4. STANDARD SCALER
# ============================================================

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaling completed.")

# ============================================================
# 5. PCA - 10 COMPONENTS
# ============================================================

print("\nApplying PCA with 10 components...")

pca = PCA(
    n_components=10
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)

variance_retained = (
    np.sum(
        pca.explained_variance_ratio_
    ) * 100
)

print(
    f"Variance retained: "
    f"{variance_retained:.4f}%"
)

print(
    "Training PCA shape:",
    X_train_pca.shape
)

print(
    "Testing PCA shape :",
    X_test_pca.shape
)

# ============================================================
# 6. TRAIN DECISION TREE
# ============================================================

print("\nTraining final Decision Tree...")

start_time = time.perf_counter()

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(
    X_train_pca,
    y_train
)

training_time = (
    time.perf_counter() - start_time
)

print(
    f"Training time: "
    f"{training_time:.4f} seconds"
)

# ============================================================
# 7. TEST MODEL
# ============================================================

print("\nTesting final model...")

start_time = time.perf_counter()

y_pred = decision_tree.predict(
    X_test_pca
)

prediction_time = (
    time.perf_counter() - start_time
)

# ============================================================
# 8. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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

# ============================================================
# 9. MODEL INFORMATION
# ============================================================

tree_depth = decision_tree.get_depth()

tree_nodes = decision_tree.tree_.node_count

leaf_nodes = decision_tree.get_n_leaves()

prediction_us_per_sample = (
    prediction_time /
    len(X_test_pca)
    * 1_000_000
)

# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print(
    f"\nPCA components     : 10"
)

print(
    f"Variance retained  : "
    f"{variance_retained:.4f}%"
)

print(
    f"\nAccuracy           : "
    f"{accuracy * 100:.4f}%"
)

print(
    f"Precision          : "
    f"{precision * 100:.4f}%"
)

print(
    f"Recall             : "
    f"{recall * 100:.4f}%"
)

print(
    f"F1 Score           : "
    f"{f1 * 100:.4f}%"
)

print(
    f"\nTree depth         : "
    f"{tree_depth}"
)

print(
    f"Tree nodes         : "
    f"{tree_nodes}"
)

print(
    f"Leaf nodes         : "
    f"{leaf_nodes}"
)

print(
    f"\nTraining time      : "
    f"{training_time:.4f} seconds"
)

print(
    f"Test inference     : "
    f"{prediction_time * 1000:.4f} ms"
)

print(
    f"Per-sample inference: "
    f"{prediction_us_per_sample:.4f} µs"
)

# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)

# ============================================================
# 13. SAVE SCALER
# ============================================================

scaler_path = (
    model_folder /
    "final_scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)

# ============================================================
# 14. SAVE PCA
# ============================================================

pca_path = (
    model_folder /
    "final_pca_10.pkl"
)

joblib.dump(
    pca,
    pca_path
)

# ============================================================
# 15. SAVE DECISION TREE
# ============================================================

tree_path = (
    model_folder /
    "final_decision_tree_10pc.pkl"
)

joblib.dump(
    decision_tree,
    tree_path
)

# ============================================================
# 16. SAVE COMPLETE PIPELINE
# ============================================================

pipeline = {
    "scaler": scaler,
    "pca": pca,
    "decision_tree": decision_tree
}

pipeline_path = (
    model_folder /
    "final_tinyml_pipeline.pkl"
)

joblib.dump(
    pipeline,
    pipeline_path
)

# ============================================================
# 17. SAVE METADATA
# ============================================================

metadata = {
    "dataset": "IoTID20",
    "original_features": 79,
    "pca_components": 10,
    "variance_retained_percent":
        float(variance_retained),

    "accuracy_percent":
        float(accuracy * 100),

    "precision_percent":
        float(precision * 100),

    "recall_percent":
        float(recall * 100),

    "f1_percent":
        float(f1 * 100),

    "tree_depth":
        int(tree_depth),

    "tree_nodes":
        int(tree_nodes),

    "leaf_nodes":
        int(leaf_nodes),

    "training_time_seconds":
        float(training_time),

    "inference_microseconds_per_sample":
        float(prediction_us_per_sample)
}

metadata_path = (
    model_folder /
    "final_model_metadata.json"
)

with open(
    metadata_path,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

# ============================================================
# 18. SAVE RESULTS CSV
# ============================================================

results = pd.DataFrame([metadata])

results.to_csv(
    output_folder /
    "final_tinyml_results.csv",
    index=False
)

# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print("\nSaved files:")

print(
    "models/final_scaler.pkl"
)

print(
    "models/final_pca_10.pkl"
)

print(
    "models/final_decision_tree_10pc.pkl"
)

print(
    "models/final_tinyml_pipeline.pkl"
)

print(
    "models/feature_names.json"
)

print(
    "models/final_model_metadata.json"
)

print(
    "outputs/final_tinyml_results.csv"
)

print("\n" + "=" * 70)
print("STEP 9 COMPLETED")
print("=" * 70)