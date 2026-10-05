import pandas as pd
import numpy as np
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
# 1. Locate project files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

# --------------------------------------------------
# 2. Load train/test data
# --------------------------------------------------

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_train = pd.read_csv(data_folder / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(data_folder / "y_test.csv").iloc[:, 0]

print("=" * 70)
print("PCA SIZE vs DECISION TREE COMPARISON")
print("=" * 70)

print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)

# --------------------------------------------------
# 3. StandardScaler
# --------------------------------------------------

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed.")

# --------------------------------------------------
# 4. PCA sizes to compare
# --------------------------------------------------

pca_sizes = [10, 18, 23, 31]

results = []

# --------------------------------------------------
# 5. Test each PCA size
# --------------------------------------------------

for n_components in pca_sizes:

    print("\n" + "-" * 70)
    print(f"Testing PCA with {n_components} components...")
    print("-" * 70)

    # PCA
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    variance = np.sum(pca.explained_variance_ratio_) * 100

    print(f"Variance retained: {variance:.4f}%")

    # --------------------------------------------------
    # Decision Tree
    # --------------------------------------------------

    print("Training Decision Tree...")

    dt = DecisionTreeClassifier(
        random_state=42
    )

    dt.fit(X_train_pca, y_train)

    # Prediction
    y_pred = dt.predict(X_test_pca)

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

    print(f"Accuracy : {accuracy * 100:.4f}%")
    print(f"Precision: {precision * 100:.4f}%")
    print(f"Recall   : {recall * 100:.4f}%")
    print(f"F1 Score : {f1 * 100:.4f}%")

    # Save result
    results.append({
        "PCA_Components": n_components,
        "Variance_Retained_%": variance,
        "Accuracy_%": accuracy * 100,
        "Precision_%": precision * 100,
        "Recall_%": recall * 100,
        "F1_%": f1 * 100
    })

# --------------------------------------------------
# 6. Display final comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# --------------------------------------------------
# 7. Find best F1
# --------------------------------------------------

best = results_df.loc[
    results_df["F1_%"].idxmax()
]

print("\n" + "=" * 70)
print("BEST PCA CONFIGURATION")
print("=" * 70)

print(
    "PCA Components       :",
    int(best["PCA_Components"])
)

print(
    "Variance Retained    :",
    f"{best['Variance_Retained_%']:.4f}%"
)

print(
    "Accuracy             :",
    f"{best['Accuracy_%']:.4f}%"
)

print(
    "Precision            :",
    f"{best['Precision_%']:.4f}%"
)

print(
    "Recall               :",
    f"{best['Recall_%']:.4f}%"
)

print(
    "F1 Score             :",
    f"{best['F1_%']:.4f}%"
)

# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

output_path = project_folder / "outputs"

output_path.mkdir(exist_ok=True)

results_df.to_csv(
    output_path / "pca_dt_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("outputs/pca_dt_comparison.csv")

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)