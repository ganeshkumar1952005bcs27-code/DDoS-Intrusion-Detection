import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Locate files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

X_path = project_folder / "data" / "X_features.csv"
y_path = project_folder / "data" / "y_labels.csv"

# --------------------------------------------------
# 2. Load X and y
# --------------------------------------------------

X = pd.read_csv(X_path)
y = pd.read_csv(y_path).iloc[:, 0]

print("=" * 60)
print("IoTID20 TRAIN / TEST SPLIT")
print("=" * 60)

print("\nOriginal data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------------------------
# 3. Split the dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

# --------------------------------------------------
# 5. Save the split datasets
# --------------------------------------------------

X_train.to_csv(project_folder / "data" / "X_train.csv", index=False)
X_test.to_csv(project_folder / "data" / "X_test.csv", index=False)

y_train.to_csv(project_folder / "data" / "y_train.csv", index=False)
y_test.to_csv(project_folder / "data" / "y_test.csv", index=False)

print("\nSaved files:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 60)