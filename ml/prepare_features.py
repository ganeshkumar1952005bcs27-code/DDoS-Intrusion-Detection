import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Locate project and cleaned dataset
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
dataset_path = project_folder / "data" / "IoTID20_cleaned.csv"

# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(dataset_path)

print("=" * 60)
print("IoTID20 FEATURE PREPARATION")
print("=" * 60)

print("\nDataset:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# --------------------------------------------------
# 3. Create binary target
# --------------------------------------------------

# Normal = 0
# Anomaly = 1

y = (df["Label"].str.lower() != "normal").astype(int)

# --------------------------------------------------
# 4. Remove non-ML columns
# --------------------------------------------------

columns_to_remove = [
    "Flow_ID",
    "Src_IP",
    "Dst_IP",
    "Timestamp",
    "Label",
    "Cat",
    "Sub_Cat"
]

X = df.drop(columns=columns_to_remove)

# --------------------------------------------------
# 5. Convert features to numeric
# --------------------------------------------------

X = X.apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------
# 6. Check result
# --------------------------------------------------

print("\nFeature matrix X:")
print("Rows    :", X.shape[0])
print("Features:", X.shape[1])

print("\nTarget y:")
print("Normal (0):", (y == 0).sum())
print("Attack (1):", (y == 1).sum())

print("\nFeature names:")
print(X.columns.tolist())

# --------------------------------------------------
# 7. Save prepared data
# --------------------------------------------------

X.to_csv(project_folder / "data" / "X_features.csv", index=False)
y.to_csv(project_folder / "data" / "y_labels.csv", index=False)

print("\nSaved:")
print("data/X_features.csv")
print("data/y_labels.csv")

print("\n" + "=" * 60)
print("FEATURE PREPARATION COMPLETED")
print("=" * 60)