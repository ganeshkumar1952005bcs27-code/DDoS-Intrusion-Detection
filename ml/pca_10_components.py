import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. Locate project folder
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"
model_folder = project_folder / "models"

model_folder.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load training and testing data
# --------------------------------------------------

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

print("=" * 60)
print("MODULE 2 - STANDARD SCALER + PCA")
print("=" * 60)

print("\nOriginal training shape:", X_train.shape)
print("Original testing shape :", X_test.shape)

# --------------------------------------------------
# 3. Standardization
# --------------------------------------------------

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed.")

# --------------------------------------------------
# 4. PCA with 10 components
# --------------------------------------------------

print("\nApplying PCA with 10 components...")

pca = PCA(n_components=10)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# --------------------------------------------------
# 5. Display explained variance
# --------------------------------------------------

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nPCA RESULTS")
print("-" * 60)

for i in range(10):
    print(
        f"PC{i + 1}: "
        f"{explained_variance[i] * 100:.4f}% "
        f"individual, "
        f"{cumulative_variance[i] * 100:.4f}% cumulative"
    )

print("\nTotal variance retained by 10 components:",
      f"{cumulative_variance[-1] * 100:.4f}%")

# --------------------------------------------------
# 6. Display new dimensions
# --------------------------------------------------

print("\nReduced training shape:", X_train_pca.shape)
print("Reduced testing shape :", X_test_pca.shape)

# --------------------------------------------------
# 7. Save transformed data
# --------------------------------------------------

train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=[f"PC{i}" for i in range(1, 11)]
)

test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=[f"PC{i}" for i in range(1, 11)]
)

train_pca_df.to_csv(
    data_folder / "X_train_pca10.csv",
    index=False
)

test_pca_df.to_csv(
    data_folder / "X_test_pca10.csv",
    index=False
)

# --------------------------------------------------
# 8. Save scaler and PCA
# --------------------------------------------------

joblib.dump(
    scaler,
    model_folder / "scaler.pkl"
)

joblib.dump(
    pca,
    model_folder / "pca_10.pkl"
)

print("\nSaved files:")
print("data/X_train_pca10.csv")
print("data/X_test_pca10.csv")
print("models/scaler.pkl")
print("models/pca_10.pkl")

print("\n" + "=" * 60)
print("PCA PROCESS COMPLETED")
print("=" * 60)