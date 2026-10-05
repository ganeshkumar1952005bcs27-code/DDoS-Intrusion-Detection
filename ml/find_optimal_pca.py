import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. Locate project folder
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

# --------------------------------------------------
# 2. Load training data
# --------------------------------------------------

X_train = pd.read_csv(data_folder / "X_train.csv")

print("=" * 60)
print("FINDING OPTIMAL PCA COMPONENTS")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)

# --------------------------------------------------
# 3. Standardize training data
# --------------------------------------------------

print("\nApplying StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Scaling completed.")

# --------------------------------------------------
# 4. Fit PCA with all components
# --------------------------------------------------

print("\nCalculating PCA...")

pca = PCA()
pca.fit(X_train_scaled)

# --------------------------------------------------
# 5. Calculate cumulative variance
# --------------------------------------------------

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# --------------------------------------------------
# 6. Find components for different thresholds
# --------------------------------------------------

thresholds = [0.90, 0.95, 0.99, 0.995]

print("\n" + "-" * 60)
print("PCA COMPONENT ANALYSIS")
print("-" * 60)

for threshold in thresholds:

    components = np.argmax(
        cumulative_variance >= threshold
    ) + 1

    retained = cumulative_variance[components - 1] * 100

    print(
        f"{threshold * 100:.1f}% variance "
        f"→ {components} components "
        f"({retained:.4f}% retained)"
    )

# --------------------------------------------------
# 7. Display first 20 components
# --------------------------------------------------

print("\n" + "-" * 60)
print("CUMULATIVE VARIANCE")
print("-" * 60)

for i in range(min(20, len(cumulative_variance))):

    print(
        f"PC{i + 1:02d}: "
        f"{cumulative_variance[i] * 100:.4f}%"
    )

# --------------------------------------------------
# 8. Find optimal number for 99%
# --------------------------------------------------

optimal_components = np.argmax(
    cumulative_variance >= 0.99
) + 1

optimal_variance = (
    cumulative_variance[optimal_components - 1] * 100
)

print("\n" + "=" * 60)
print("RECOMMENDED PCA SIZE")
print("=" * 60)

print("Original features       :", X_train.shape[1])
print("Recommended components  :", optimal_components)
print(
    "Variance retained       :",
    f"{optimal_variance:.4f}%"
)

print("=" * 60)